"""Read-only inventory and bounded, isolated life10 experiments over existing SSH aliases.

Nothing starts until ``launch`` is explicitly invoked. ``prepare`` freezes Python
source and an experiment manifest; it does not copy credentials, data, or live runs.
Examples::

    python scripts/life10_cluster.py inventory --out runs/life10/inventory.json
    python scripts/life10_cluster.py prepare v10-smoke --mode smoke
    python scripts/life10_cluster.py launch v10-smoke
    python scripts/life10_cluster.py status v10-smoke
    python scripts/life10_cluster.py collect v10-smoke

All remote work stays inside ~/life10/EXPERIMENT. Existing directories are refused.
No service, tmux session, or existing process is stopped. ROCm uses torch's cuda API.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("adler40", "knecht24", "falke64", "specht32")
PROFILES = ("normal", "dark", "no_template", "no_catalysis", "no_compartments")
MIB = 1024 ** 2

# Run through the hosts' existing interpreter, not system Python (which lacks torch).
PROBE = r'''
import collections,json,os,pathlib,platform,shutil,subprocess,time
def cpu_ticks():
    vals=[int(x) for x in pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return sum(vals),vals[3]+vals[4]
a=cpu_ticks(); time.sleep(.2); b=cpu_ticks()
idle=(b[1]-a[1])/max(1,b[0]-a[0])
procs=collections.Counter()
for p in pathlib.Path('/proc').iterdir():
    if p.name.isdecimal():
        try: procs[(p/'comm').read_text().strip()]+=1
        except (OSError,UnicodeError): pass
interesting={k:v for k,v in procs.items() if any(w in k for w in ('python','homunculi','ggml','llama','claude'))}
ram={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())}
report={'host':platform.node(),'platform':platform.platform(),'python':platform.python_version(),
        'cpu_count':os.cpu_count(),'cpu_idle_fraction':round(idle,3),'memory_available_bytes':ram['MemAvailable'],
        'disk_free_bytes':shutil.disk_usage(pathlib.Path.home()).free,'process_names':interesting,'gpus':[]}
report['cpu_name']=next((line.split(':',1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
try:
    lines=subprocess.run(['lspci','-nn'],capture_output=True,text=True,timeout=5).stdout.splitlines()
    report['physical_gpu_buses']=[line for line in lines if 'VGA compatible' in line or 'Display controller' in line]
except (OSError,subprocess.TimeoutExpired): report['physical_gpu_buses']=[]
try:
    import torch
    report.update(torch=torch.__version__,cuda=torch.version.cuda,hip=getattr(torch.version,'hip',None))
    amd_physical=[]
    for card in pathlib.Path('/sys/class/drm').glob('card*'):
        if not card.name[4:].isdecimal(): continue
        device=card/'device'
        try:
            if (device/'vendor').read_text().strip()!='0x1002': continue
            total=int((device/'mem_info_vram_total').read_text())
            used=int((device/'mem_info_vram_used').read_text())
            # The Raphael iGPU has a small VRAM aperture, despite torch advertising system RAM.
            if total>1024**3:
                amd_physical.append({'pci':device.resolve().name,'total_bytes':total,'free_bytes':max(0,total-used)})
        except (OSError,ValueError): pass
    amd_physical.sort(key=lambda item:item['pci'])
    amd_cursor=0
    for i in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(i)
        integrated=any(s in props.name.lower() for s in ('ryzen','integrated','uhd','intel'))
        try:
            with torch.cuda.device(i): free,total=torch.cuda.mem_get_info()
        except Exception: free,total=0,props.total_memory
        item={'index':i,'name':props.name,'total_bytes':total,'free_bytes':free,
              'discrete':not integrated,'gcn_arch':getattr(props,'gcnArchName',None)}
        if report['hip'] and not integrated and amd_cursor<len(amd_physical):
            physical=amd_physical[amd_cursor]; amd_cursor+=1
            item['runtime_free_bytes']=free; item['physical_vram_free_bytes']=physical['free_bytes']
            item['pci']=physical['pci']; item['free_bytes']=min(free,physical['free_bytes'])
        report['gpus'].append(item)
except Exception as exc: report['torch_error']=str(exc)[:300]
try:
    report['sessions']=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True,timeout=5).stdout.splitlines()
except (OSError,subprocess.TimeoutExpired): report['sessions']=[]
print(json.dumps(report,sort_keys=True))
'''

# This wrapper supervises only its own children. No remote shell concatenates user args.
WORKER = r'''
import concurrent.futures,hashlib,json,os,pathlib,shutil,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
core={k:v for k,v in manifest.items() if k!='manifest_sha256'}
actual=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if actual!=manifest['manifest_sha256']: raise SystemExit('manifest checksum mismatch')
code=root/'code'
for filename,digest in manifest['source_files'].items():
    if hashlib.sha256((code/filename).read_bytes()).hexdigest()!=digest:
        raise SystemExit('source checksum mismatch: '+filename)
if manifest.get('chains_sha256'):
    if hashlib.sha256((root/'chains.json').read_bytes()).hexdigest()!=manifest['chains_sha256']:
        raise SystemExit('chain input checksum mismatch')
host=os.uname().nodename
lanes=manifest['lanes'].get(host,[])
if not lanes: raise SystemExit('no lanes for host '+host)
def write(path,data):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)); temp.replace(path)
def size():
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file(): total+=p.stat().st_size
        except OSError: pass
    return total
def lane(job):
    name=job['name']; folder=root/'out'/name; folder.mkdir(parents=True,exist_ok=False)
    state={'lane':name,'device':job['device'],'seeds':job['seeds'],'profiles':[],'status':'running'}
    status=folder/'status.json'; write(status,state)
    for profile in manifest['profiles']:
        free=shutil.disk_usage(root).free
        if size()>manifest['max_output_mb']*1024**2 or free<manifest['min_disk_free_mb']*1024**2:
            state['status']='output_budget'; write(status,state); return state
        out=folder/profile
        args=['-m','haishool.life10','run','--seeds',*[str(n) for n in job['seeds']],
              '--device',job['device'],'--steps',str(manifest['steps']),'--out',str(out),
              '--threads',str(job['threads']),'--profile',profile,
              '--grid',str(manifest['grid']),'--samples',str(manifest['samples']),
              '--repeat',str(job['repeat'])]
        if manifest.get('chains_sha256'): args+=['--chains',str(root/'chains.json')]
        # Enforce a modest torch allocator ceiling. CPU uses the same immutable module.
        driver='import sys,runpy,torch; '
        if job['device'].startswith('cuda'):
            driver+=('d=torch.device('+repr(job['device'])+'); '
                     'torch.cuda.set_per_process_memory_fraction(min(.95,'+
                     str(manifest['gpu_memory_mb']*1024**2)+'/torch.cuda.get_device_properties(d).total_memory),d); ')
        driver+='sys.argv='+repr(args[1:])+'; runpy.run_module("haishool.life10",run_name="__main__")'
        env=dict(os.environ,OMP_NUM_THREADS=str(job['threads']),MKL_NUM_THREADS=str(job['threads']),
                 PYTHONPATH=str(code),PYTHONUNBUFFERED='1')
        start=time.time(); log=folder/(profile+'.log')
        with log.open('wb') as stream:
            proc=subprocess.Popen(['nice','-n','10',sys.executable,'-c',driver],cwd=code,env=env,
                                  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            deadline=start+manifest['lane_timeout_seconds']; reason=None
            while proc.poll() is None:
                time.sleep(1)
                if size()>manifest['max_output_mb']*1024**2: reason='output_budget'
                elif shutil.disk_usage(root).free<manifest['min_disk_free_mb']*1024**2: reason='disk_reserve'
                elif time.time()>deadline: reason='timeout'
                if reason:
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    break
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-4096)); tail=stream.read().decode(errors='replace')
        row={'profile':profile,'exit_code':proc.returncode,'seconds':round(time.time()-start,2)}
        if reason: row['reason']=reason
        if proc.returncode and ('out of memory' in tail.lower() or 'memoryerror' in tail.lower()):
            row['reason']='skipped_memory_limit'
        state['profiles'].append(row); write(status,state)
        if reason: state['status']=reason; write(status,state); return state
    state['status']='complete' if all(p['exit_code']==0 for p in state['profiles']) else 'completed_with_skips_or_errors'
    write(status,state); return state
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        results=list(pool.map(lane,lanes))
    write(root/'status.json',{'status':'finished','lanes':results,'finished_unix':time.time()})
except Exception as exc:
    write(root/'status.json',{'status':'error','error':str(exc)[:1000]}); raise
'''


def valid_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) or value in (".", ".."):
        raise ValueError("experiment name must be 1-80 letters, digits, dots, underscores or hyphens")
    return value


def host_arg(value: str) -> str:
    if value not in HOSTS:
        raise ValueError(f"unknown host {value!r}; allowed: {HOSTS}")
    return value


def ssh(host: str, command: str, payload: bytes | None = None, timeout: int = 60) -> bytes:
    host_arg(host)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{host}: {result.stderr.decode(errors='replace')[-1000:]}")
    return result.stdout


def probe_host(host: str) -> dict:
    try:
        raw = ssh(host, '"$HOME/homunculi/.venv/bin/python" -', PROBE.encode())
        # ROCm may emit harmless driver diagnostics before the JSON line.
        row = json.loads(raw.decode().splitlines()[-1])
        row["alias"] = host
        return row
    except Exception as exc:
        return {"alias": host, "error": str(exc)}


def inventory(hosts: list[str]) -> dict:
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        records = list(pool.map(probe_host, hosts))
    import platform
    local = {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
             "cpu_count": os.cpu_count(), "disk_free_bytes": __import__("shutil").disk_usage(ROOT).free,
             "execution": "local CPU launches are separate; preserve any active Opus work"}
    try:
        import torch
        local.update(torch=torch.__version__, cuda=torch.version.cuda, hip=getattr(torch.version, "hip", None),
                     cuda_available=torch.cuda.is_available())
    except ImportError:
        local["torch_error"] = "torch unavailable"
    return {"schema": "life10-inventory-v1", "utc": datetime.now(timezone.utc).isoformat(), "hosts": records, "local": local}


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def choose_lanes(records: list[dict], start_seed: int, worlds: int, memory_mb: int,
                 seeds: list[int] | None = None, repeat: int = 0) -> dict:
    """Reserve feeder/service threads; include occupied discrete GPUs only if budget fits."""
    result = {}
    seed = start_seed
    for record in records:
        if "error" in record or "torch_error" in record:
            continue
        alias = record["alias"]
        GPUs = [g for g in record["gpus"] if g["discrete"] and g["free_bytes"] >= (memory_mb + 256)*MIB]
        jobs = [{"name": f"gpu{g['index']}", "device": f"cuda:{g['index']}", "threads": 1} for g in GPUs]
        jobs.append({"name": "cpu", "device": "cpu", "threads": max(1, record["cpu_count"]-len(GPUs)-2)})
        for job in jobs:
            job["seeds"] = list(seeds) if seeds is not None else list(range(seed, seed+worlds))
            job["repeat"] = repeat
            repeat += 1
            seed += worlds
        result[alias] = jobs
    return result


def snapshot_sources(root: Path) -> tuple[bytes, dict]:
    weights = root / "data" / "truth-v5" / "atomic_weights.json"
    if not weights.is_file():
        raise ValueError(f"required atomic weight data is missing: {weights}")
    paths = [root / "haishool" / "__init__.py", weights]
    for package in ("life10", "cosmos", "evo", "truth"):
        directory = root / "haishool" / package
        if not directory.is_dir():
            raise ValueError(f"source package is missing: {directory}")
        paths.extend(sorted(directory.rglob("*.py")))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"refusing symlink in source snapshot: {path}")
    content = {p.relative_to(root).as_posix(): p.read_bytes() for p in paths}
    if any(p.read_bytes() != content[p.relative_to(root).as_posix()] for p in paths):
        raise ValueError("source changed while being frozen; wait for the active edit to finish")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(content.items())}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in sorted(content.items()):
            item = tarfile.TarInfo(name); item.size = len(data); item.mode = 0o644; item.mtime = 0
            archive.addfile(item, io.BytesIO(data))
    return gzip.compress(stream.getvalue(), mtime=0), hashes


def experiment_path(name: str, output_root: Path) -> Path:
    return output_root.resolve() / valid_name(name)


def prepare(args) -> dict:
    folder = experiment_path(args.experiment, args.output_root)
    if folder.exists():
        raise ValueError(f"experiment already exists: {folder}")
    inv = inventory(args.hosts)
    source, source_files = snapshot_sources(ROOT)
    worlds = args.worlds or (2 if args.mode == "smoke" else 16)
    steps = args.steps or (4 if args.mode == "smoke" else 300)
    chains = None
    if args.chains:
        chains = args.chains.read_bytes()
        if not isinstance(json.loads(chains), list):
            raise ValueError("cached chains must be a JSON list")
    manifest = {"schema": "life10-cluster-v1", "experiment": args.experiment,
                "created_utc": inv["utc"], "mode": args.mode, "steps": steps,
                "profiles": args.profiles, "source_files": source_files,
                "source_sha256": hashlib.sha256(source).hexdigest(), "inventory": inv,
                "worker_sha256": hashlib.sha256(WORKER.encode()).hexdigest(),
                "lanes": choose_lanes(inv["hosts"], args.start_seed, worlds, args.gpu_memory_mb, args.seeds, args.repeat),
                "grid": args.grid, "samples": args.samples,
                "chains_sha256": None if chains is None else hashlib.sha256(chains).hexdigest(),
                "max_output_mb": args.max_output_mb, "min_disk_free_mb": args.min_disk_free_mb,
                "gpu_memory_mb": args.gpu_memory_mb, "lane_timeout_seconds": args.timeout,
                "seed_policy": "explicit seeds shared across lanes or disjoint sequential seeds; distinct repeat per lane; same seed+repeat across profiles",
                "execution_policy": "bounded jobs; no service stops; no outcome filtering or reseeding"}
    if not manifest["lanes"]:
        raise ValueError("no reachable torch-enabled hosts")
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    folder.mkdir(parents=True)
    (folder / "source.tar.gz").write_bytes(source)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (folder / "worker.py").write_bytes(WORKER.encode())
    if chains is not None: (folder / "chains.json").write_bytes(chains)
    return {"prepared": str(folder), "manifest_sha256": manifest["manifest_sha256"], "lanes": manifest["lanes"]}


def load_experiment(args) -> tuple[Path, dict]:
    folder = experiment_path(args.experiment, args.output_root)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if hashlib.sha256(canonical(core)).hexdigest() != manifest["manifest_sha256"]:
        raise ValueError("immutable manifest checksum mismatch")
    if hashlib.sha256((folder / "source.tar.gz").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("source archive checksum mismatch")
    if hashlib.sha256((folder / "worker.py").read_bytes()).hexdigest() != manifest["worker_sha256"]:
        raise ValueError("worker checksum mismatch")
    if manifest.get("chains_sha256") and hashlib.sha256((folder / "chains.json").read_bytes()).hexdigest() != manifest["chains_sha256"]:
        raise ValueError("chain input checksum mismatch")
    return folder, manifest


def launch(args) -> dict:
    folder, manifest = load_experiment(args)
    results = {}
    for host in manifest["lanes"]:
        # A fresh exclusive directory is created before upload. Failure never touches it again.
        remote = f"life10/{valid_name(args.experiment)}"
        bootstrap = f'''import pathlib,tarfile,sys,io
r=pathlib.Path.home()/{remote!r}
r.mkdir(parents=True,exist_ok=False)
print(str(r))
'''
        try:
            target = ssh(host, '"$HOME/homunculi/.venv/bin/python" -', bootstrap.encode()).decode().strip()
            names = ["manifest.json", "source.tar.gz", "worker.py"]
            if manifest.get("chains_sha256"): names.append("chains.json")
            for name in names:
                ssh(host, "cat > " + shlex.quote(target+"/"+name), (folder / name).read_bytes())
            check = f'''import pathlib,hashlib,json,tarfile,subprocess,sys,os
r=pathlib.Path({target!r}); m=json.loads((r/'manifest.json').read_text())
assert hashlib.sha256((r/'source.tar.gz').read_bytes()).hexdigest()==m['source_sha256']
assert hashlib.sha256((r/'worker.py').read_bytes()).hexdigest()==m['worker_sha256']
code=r/'code'; code.mkdir()
with tarfile.open(r/'source.tar.gz') as a:
    for p in a.getmembers():
        assert p.isfile() and (p.name.startswith('haishool/') or p.name=='data/truth-v5/atomic_weights.json') and '..' not in pathlib.PurePosixPath(p.name).parts
    a.extractall(code)
with (r/'launcher.log').open('wb') as log:
    p=subprocess.Popen([sys.executable,str(r/'worker.py')],cwd=r,stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'launcher.pid').write_text(str(p.pid))
print(json.dumps({{'pid':p.pid,'path':str(r)}}))
'''
            results[host] = json.loads(ssh(host, '"$HOME/homunculi/.venv/bin/python" -', check.encode()).decode())
        except Exception as exc:
            results[host] = {"error": str(exc)}
    (folder / "launch-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def status(args) -> dict:
    _, manifest = load_experiment(args)
    results = {}
    for host in manifest["lanes"]:
        script = f'''import json,pathlib,os
r=pathlib.Path.home()/'life10'/{args.experiment!r}
result={{'path':str(r),'exists':r.exists(),'lanes':[]}}
if (r/'launcher.pid').exists():
    try: os.kill(int((r/'launcher.pid').read_text()),0); result['launcher_running']=True
    except ProcessLookupError: result['launcher_running']=False
if (r/'status.json').exists(): result['final']=json.loads((r/'status.json').read_text())
for p in sorted(r.glob('out/*/status.json')): result['lanes'].append(json.loads(p.read_text()))
print(json.dumps(result))
'''
        try: results[host] = json.loads(ssh(host, '"$HOME/homunculi/.venv/bin/python" -', script.encode()).decode())
        except Exception as exc: results[host] = {"error": str(exc)}
    return results


def collect(args) -> dict:
    folder, manifest = load_experiment(args)
    results = {}
    for host in manifest["lanes"]:
        dest = folder / host
        if dest.exists():
            results[host] = {"skipped": "local collection exists; use a fresh destination"}; continue
        remote = f"life10/{valid_name(args.experiment)}"
        command = 'tar -czf - -C "$HOME/' + remote + '" manifest.json status.json out'
        try:
            payload = ssh(host, command, timeout=120)
            if len(payload) > manifest["max_output_mb"]*MIB:
                raise ValueError("collection exceeds output budget")
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                for item in archive.getmembers():
                    path = Path(item.name)
                    if path.is_absolute() or ".." in path.parts or item.issym() or item.islnk():
                        raise ValueError("unsafe archive member")
                dest.mkdir()
                archive.extractall(dest)
            results[host] = {"collected": str(dest), "compressed_bytes": len(payload)}
        except Exception as exc: results[host] = {"error": str(exc)}
    return results


def positive(value: str) -> int:
    number = int(value)
    if number <= 0: raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "life10")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory"); inv.add_argument("--hosts", nargs="+", choices=HOSTS, default=list(HOSTS)); inv.add_argument("--out", type=Path)
    prep = sub.add_parser("prepare"); prep.add_argument("experiment", type=valid_name)
    prep.add_argument("--hosts", nargs="+", choices=HOSTS, default=list(HOSTS))
    prep.add_argument("--mode", choices=("smoke", "search"), default="smoke")
    prep.add_argument("--worlds", type=positive); prep.add_argument("--steps", type=positive)
    prep.add_argument("--start-seed", type=int, default=0)
    prep.add_argument("--seeds", nargs="+", type=int, help="share these cosmic seeds across all lanes")
    prep.add_argument("--repeat", type=int, default=0, help="first independent replicate number")
    prep.add_argument("--chains", type=Path, help="immutable precomputed chain JSON list")
    prep.add_argument("--grid", type=positive, default=4); prep.add_argument("--samples", type=positive, default=10000)
    prep.add_argument("--profiles", nargs="+", choices=PROFILES, default=["normal"])
    prep.add_argument("--gpu-memory-mb", type=positive, default=256)
    prep.add_argument("--max-output-mb", type=positive, default=512)
    prep.add_argument("--min-disk-free-mb", type=positive, default=2048)
    prep.add_argument("--timeout", type=positive, default=3600)
    for action in ("launch", "status", "collect"):
        sub.add_parser(action).add_argument("experiment", type=valid_name)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory(args.hosts)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else: result = globals()[args.command](args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, str(exc)+"\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
