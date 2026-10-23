from graphrag.extraction.parse import Triple
from graphrag.ingest.cache import ExtractionCache
from graphrag.ratelimit import RateLimiter


def test_cache_persists_and_round_trips(tmp_path):
    path = tmp_path / "c" / "cache.json"
    cache = ExtractionCache(path)
    t = Triple("A", "inhibits", "B", "a", "b")
    key = ExtractionCache.key("fp1", "hash1")
    assert cache.get(key) is None
    cache.put(key, [t])
    cache.put(ExtractionCache.key("fp1", "empty"), [])
    cache.save()
    reloaded = ExtractionCache(path)
    assert reloaded.get(key) == [t]
    assert reloaded.get(ExtractionCache.key("fp1", "empty")) == []  # empty != miss
    assert reloaded.get(ExtractionCache.key("fp2", "hash1")) is None


def test_rate_limiter_spaces_calls():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    rl = RateLimiter(30, clock=lambda: now[0], sleep=sleep)  # 2s interval
    rl.wait()
    now[0] += 0.5
    rl.wait()
    now[0] += 5.0
    rl.wait()
    assert slept == [1.5]
