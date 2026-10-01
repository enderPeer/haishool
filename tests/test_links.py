"""Round-4 chat routing (no model needed): python -m pytest tests"""

from haishool.links import Graph, node_in, path_text, relation_for, relation_sentence, two_nodes, view_for

G = Graph(rels={"turkey": ["borders", "has_city"], "israel": ["has_capital", "borders"],
                "atlantic_records": ["founded_by", "signed"], "led_zeppelin": ["has_member"],
                "abraham": ["father_figure_of"], "judaism": ["holds_holy"]},
          views={("israel", "has_capital"): ["israel", "united_states"]},
          all_views={"israel", "palestine", "judaism", "united_states"},
          relation_names={"borders", "has_city", "has_capital", "founded_by", "signed", "has_member",
                          "father_figure_of", "holds_holy", "founded"})


def test_relation_routing():
    assert relation_for("what borders turkey", "turkey", G) == "borders"
    assert relation_for("what is the capital of israel", "israel", G) == "has_capital"
    assert relation_for("who founded atlantic records", "atlantic_records", G) == "founded_by"
    assert relation_for("who is in led zeppelin", "led_zeppelin", G) == "has_member"
    assert relation_for("what is turkey", "turkey", G) is None
    assert view_for("what is the capital of israel according to palestine", G) == "palestine"
    assert node_in("what is holy to the jews", G) == "judaism"
    assert two_nodes("how is abraham connected to led zeppelin", G) == ("abraham", "led_zeppelin")


def test_sentences():
    assert relation_sentence("turkey", "borders", "greece syria", G) == "Turkey borders Greece and Syria."
    text = relation_sentence("israel", "has_capital", "jerusalem contested", G)
    assert text.startswith("Israel has the capital Jerusalem. This is contested") and "United States" in text
    assert relation_sentence("abraham", "founded", "judaism according_to judaism", G) == \
        "Abraham founded Judaism (according to Judaism)."
    assert path_text("abraham father_figure_of judaism according_to judaism", "1", G) == \
        "Abraham → father figure of → Judaism (according to Judaism). That is 1 hop."


def test_yes_no_follows_the_links():
    from haishool.links import edge_truth, yes_no, yes_no_sentence
    g = Graph(rels={"turkey": ["borders"], "greece": ["borders"], "israel": ["borders"]},
              relation_names={"borders"},
              edges={("turkey", "borders", "greece"): ["fact"], ("greece", "borders", "turkey"): ["fact"]})
    assert yes_no("does turkey border greece", g) == ("turkey", "borders", "greece")
    assert yes_no("what borders turkey", g) is None
    assert yes_no_sentence("turkey", "borders", "greece", edge_truth(g, "turkey", "borders", "greece"), "yes") == \
        "Yes: Turkey borders Greece."
    no = yes_no_sentence("turkey", "borders", "israel", edge_truth(g, "turkey", "borders", "israel"), "yes")
    assert no.startswith("Not in what I learned: I have no link saying that Turkey borders Israel.") and "guessed otherwise" in no
