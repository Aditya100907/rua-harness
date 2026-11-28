"""Twin-distractor data (policy/twin_data.py): XML surgery, state mapping, virtual task paths.
No simulator or dataset needed except where marked."""
import numpy as np
import pytest

from policy import paths
from policy import twin_data as td

XML = """<mujoco><worldbody>
  <body name="robot"><joint name="j0" type="hinge"/><geom name="r_g" size="0.1"/></body>
  <body name="obj_main" pos="1 0 0"><geom name="obj_g0" size="0.05" mesh="obj_M_vis"/>
    <body name="obj_M_Lid"><geom name="obj_g1" size="0.02"/></body>
    <joint name="obj_joint0" type="free"/><site name="obj_default_site"/></body>
  <body name="door"><joint name="d0" type="slide"/><geom name="d_g" size="0.1"/></body>
</worldbody></mujoco>"""


def test_twin_xml_copies_body_with_new_names():
    import xml.etree.ElementTree as ET
    out = td.twin_xml(XML, [0.5, 0.2, 0.9], [1, 0, 0, 0])
    root = ET.fromstring(out)
    wb = root.find("worldbody")
    bodies = [b.get("name") for b in wb if b.tag == "body"]
    assert bodies == ["robot", "obj_main", "door", "twin_main"]        # appended last
    tw = wb[-1]
    names = [e.get("name") for e in tw.iter() if e.get("name")]
    assert names == ["twin_main", "twin_g0", "twin_M_Lid", "twin_g1", "twin_joint0", "twin_default_site"]
    assert tw.find("geom").get("mesh") == "obj_M_vis"                 # mesh reference shared, not renamed
    assert [float(v) for v in tw.get("pos").split()] == [0.5, 0.2, 0.9]
    # the source body is untouched
    src = next(b for b in wb if b.get("name") == "obj_main")
    assert src.get("pos") == "1 0 0"


def test_state_map_by_joint_name():
    # stock: j0 hinge (q0, v0), obj free (q1..7, v1..6), d0 slide (q8, v7)
    old = {"j0": (0, 1, 0, 1), "obj_joint0": (1, 7, 1, 6), "d0": (8, 1, 7, 1)}
    # new layout with the twin inserted in the middle (worst case) to check name-based mapping
    new = {"j0": (0, 1, 0, 1), "obj_joint0": (1, 7, 1, 6), "twin_joint0": (8, 7, 7, 6), "d0": (15, 1, 13, 1)}
    twin_q = np.array([5, 6, 7, 1, 0, 0, 0.0])
    m = td.StateMap(old, new, 9, 8, 16, 14, twin_q, "twin_joint0")
    q = np.arange(9) + 100.0
    v = np.arange(8) + 200.0
    s = np.concatenate([[3.5], q, v])
    out = m(s)
    assert out[0] == 3.5
    q2, v2 = out[1:17], out[17:]
    assert q2[0] == 100 and np.all(q2[1:8] == q[1:8]) and q2[15] == 108
    assert np.all(q2[8:15] == twin_q)
    assert v2[0] == 200 and np.all(v2[1:7] == v[1:7]) and v2[13] == 207
    assert np.all(v2[7:13] == 0)


def test_state_map_rejects_missing_joint():
    old = {"j0": (0, 1, 0, 1), "gone": (1, 1, 1, 1)}
    new = {"j0": (0, 1, 0, 1), "twin_joint0": (1, 7, 1, 6)}
    with pytest.raises(ValueError):
        td.StateMap(old, new, 2, 2, 8, 7, np.zeros(7), "twin_joint0")


def test_virtual_task_paths():
    assert paths.source_task(td.TASK) == td.SOURCE
    assert paths.source_task("OpenDrawer") == "OpenDrawer"
    try:
        src = paths.task_dir(td.SOURCE)
    except FileNotFoundError:
        pytest.skip("no PickPlaceCounterToCabinet demos on this machine")
    assert paths.task_dir(td.TASK) == src


def test_yaw_quat_composition_keeps_upright():
    q0 = np.array([1.0, 0, 0, 0])
    q = td._qmul(td._yaw_quat(np.pi / 2), q0)
    assert np.allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
