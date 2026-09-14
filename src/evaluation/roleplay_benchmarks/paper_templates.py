"""Extract RMTBench Appendix E from a hash-pinned paper, without vendoring it."""
from html.parser import HTMLParser
import re

from .sources import ROOT, digest, download

URL = "https://arxiv.org/html/2507.20352v2"
SHA256 = "2a32aa8844a96b112e29e2273027661465918cc704457d467bdc53d72b74bd61"
DIMENSIONS = {"EE": "Emotional Expression", "EC": "Emotional Comprehension",
              "PA": "Plot Advancement", "CU": "Character Consistency"}


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = {}
        self.active = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table" and attrs.get("id", "").startswith("A5.T"):
            self.active = attrs["id"].split(".")[1]
            self.tables[self.active] = []

    def handle_endtag(self, tag):
        if self.active and tag in ("tr", "td"):
            self.tables[self.active].append("\n")
        if tag == "table":
            self.active = None

    def handle_data(self, text):
        if self.active:
            self.tables[self.active].append(text)


def load_templates():
    path = ROOT / "data/external/roleplay_benchmarks/rmtbench-paper/2507.20352v2.html"
    if not path.exists():
        download(URL, path)
    if digest(path) != SHA256:
        raise ValueError("RMTBench paper HTML differs from pinned hash; review source before updating")
    parser = Tables()
    parser.feed(path.read_text(encoding="utf-8"))
    tables = {k: "\n".join(line.strip() for line in "".join(v).splitlines() if line.strip())
              for k, v in parser.tables.items()}
    rubric = tables["T12"]
    dimensions = {}
    for key, name in DIMENSIONS.items():
        match = re.search(re.escape(name) + r"[’'\"]\s*:\s*\"\"\"(.*?)\"\"\"", rubric, re.S)
        if not match:
            raise ValueError(f"Cannot extract published rubric: {key}")
        dimensions[key] = match.group(1).strip()
    old = "<|Answer Start|>\n{response}\n<|Answer End|>"
    if tables["T10"].count(old) != 1:
        raise ValueError("UPA Answer template anchor changed")
    return {"source": URL, "source_sha256": SHA256, "dimensions": dimensions,
            "dimension_names": DIMENSIONS, "quality": tables["T11"], "CM": tables["T8"],
            "UPA": tables["T10"].replace(old, old.replace("{response}", "{answer}")),
            "changes": ["HTML table whitespace normalized", "Table 10 Answer placeholder corrected from response to answer"]}
