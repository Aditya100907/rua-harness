"""Pure parts of the open-model commander (no simulator, no server)."""
import json

from policy.commander.open_model import check_command, schema, strip_old_images, to_sim_pixel


def test_coords():
    assert to_sim_pixel([0, 0], "norm1000", 512) == [0, 0]
    assert to_sim_pixel([500, 1000], "norm1000", 512) == [128, 255]
    assert to_sim_pixel([511, 256], "pixel", 512) == [255, 128]
    assert to_sim_pixel([1200, 10], "norm1000", 512) is None


def test_check_command():
    c, e = check_command({"skill": "open", "object": {"name": "drawer", "camera": "right", "uv": [500, 500]},
                          "target": {"name": "x", "camera": "left", "uv": [1, 1]}}, "norm1000", 512)
    assert e is None and "target" not in c and c["object"]["uv"] == [128, 128]
    _, e = check_command({"skill": "pick_place", "object": {"name": "apple", "camera": "left", "uv": [1, 1]}},
                         "norm1000", 512)
    assert "target" in e
    _, e = check_command({"skill": "fly"}, "norm1000", 512)
    assert e


def test_schema_and_strip():
    s = schema()
    assert "open" in s["properties"]["command"]["properties"]["skill"]["enum"]
    assert "$ref" not in json.dumps(s)
    img = {"type": "image_url", "image_url": {"url": "x"}}
    msgs = [{"role": "system", "content": "s"}] + [{"role": "user", "content": [{"type": "text", "text": "t"}, img]}
                                                   for _ in range(3)]
    strip_old_images(msgs, 1)
    assert [any(p["type"] == "image_url" for p in m["content"]) for m in msgs[1:]] == [False, False, True]
