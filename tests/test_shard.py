import pytest

from vn_legal_graph.shard import parse_shard, shard_suffix, take_shard


def test_shards_are_disjoint_and_cover_everything():
    items = list(range(10))
    parts = [take_shard(items, f"{k}/3") for k in (1, 2, 3)]
    assert parts == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(x for p in parts for x in p) == items


def test_suffix_and_errors():
    assert shard_suffix("") == shard_suffix("1/1") == ""
    assert shard_suffix("2/4") == "_phan2-4"
    for bad in ("0/2", "3/2", "a/b", "2"):
        with pytest.raises(ValueError):
            parse_shard(bad)
