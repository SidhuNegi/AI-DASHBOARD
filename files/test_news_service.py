import unittest
from datetime import datetime, timedelta, timezone

from services.news_service import (
    build_fresh_news_snapshot,
    build_gemini_news_context,
    deduplicate_headlines,
    fetch_raw_news,
    filter_fresh_news,
    format_relative_age,
    google_news_rss_url,
    google_news_when_window,
    normalize_headline,
    parse_feed_timestamp,
    prune_classifications,
    prune_pending_alerts,
    validate_news_snapshot,
    verified_items_for_gemini,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 18, 16, 0, tzinfo=UTC)


def parsed(dt):
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0, 0, 0)


def entry(title="Nifty rises on strong bank stocks - Example News", published=None,
          updated=None, link="https://example.com/story", **extra):
    data = {"title": title, "link": link}
    if published is not ...:
        data["published_parsed"] = parsed(published) if isinstance(published, datetime) else published
    if updated is not ...:
        data["updated_parsed"] = parsed(updated) if isinstance(updated, datetime) else updated
    data.update(extra)
    return data


class TimestampValidationTests(unittest.TestCase):
    def test_ten_minutes_old_is_accepted(self):
        items, stats = filter_fresh_news([entry(published=NOW - timedelta(minutes=10))], now_utc=NOW)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["age_minutes"], 10)
        self.assertEqual(items[0]["timestamp_source"], "published_parsed")

    def test_seventeen_hours_old_is_accepted(self):
        items, _ = filter_fresh_news([entry(published=NOW - timedelta(hours=17))], now_utc=NOW)
        self.assertEqual(len(items), 1)

    def test_over_eighteen_hours_old_is_rejected(self):
        items, stats = filter_fresh_news([entry(published=NOW - timedelta(hours=18, seconds=1))], now_utc=NOW)
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_stale_count"], 1)

    def test_old_publication_cannot_be_rescued_by_recent_update(self):
        items, stats = filter_fresh_news([entry(
            published=NOW - timedelta(days=14), updated=NOW - timedelta(minutes=44)
        )], now_utc=NOW, allow_updated_fallback=True)
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_stale_count"], 1)

    def test_missing_publication_is_rejected_under_strict_default(self):
        items, stats = filter_fresh_news([entry(published=None, updated=NOW - timedelta(minutes=10))], now_utc=NOW)
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_missing_timestamp_count"], 1)

    def test_malformed_publication_timestamp_is_rejected(self):
        items, stats = filter_fresh_news([entry(published=(2026, 14, 45, 99, 0, 0))], now_utc=NOW)
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_missing_timestamp_count"], 1)

    def test_explicit_updated_timestamp_fallback_is_labeled(self):
        items, _ = filter_fresh_news(
            [entry(published=None, updated=NOW - timedelta(minutes=20))],
            now_utc=NOW, allow_updated_fallback=True,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["timestamp_source"], "updated_parsed")

    def test_neither_usable_timestamp_is_rejected(self):
        items, stats = filter_fresh_news([entry(published=None, updated=None)], now_utc=NOW)
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_missing_timestamp_count"], 1)

    def test_future_inside_skew_is_accepted_but_not_shown_as_newly_published(self):
        future = NOW + timedelta(minutes=3)
        items, _ = filter_fresh_news([entry(published=future)], now_utc=NOW, future_skew_minutes=5)
        self.assertEqual(len(items), 1)
        self.assertIn("clock skew", format_relative_age(items[0]["published_at_utc"], now_utc=NOW))

    def test_future_beyond_skew_is_rejected(self):
        items, stats = filter_fresh_news(
            [entry(published=NOW + timedelta(minutes=6))], now_utc=NOW, future_skew_minutes=5
        )
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_future_timestamp_count"], 1)

    def test_timezone_aware_datetime_is_normalized_to_utc(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        local = datetime(2026, 10, 18, 21, 20, tzinfo=ist)
        parsed_value = parse_feed_timestamp(local)
        self.assertEqual(parsed_value, datetime(2026, 10, 18, 15, 50, tzinfo=UTC))

    def test_relative_age_uses_published_timestamp_and_injected_clock(self):
        published = NOW - timedelta(hours=3, minutes=20)
        self.assertEqual(format_relative_age(published.isoformat(), now_utc=NOW), "3h ago")
        self.assertEqual(format_relative_age((NOW - timedelta(minutes=12)).isoformat(), now_utc=NOW), "12m ago")
        self.assertEqual(format_relative_age("not-a-timestamp", now_utc=NOW), "timestamp unavailable")

    def test_malformed_entry_does_not_crash_batch(self):
        raw = [None, {"title": "bad timestamp", "published_parsed": (2026,), "link": "https://example.com"},
               entry(published=NOW - timedelta(minutes=10))]
        items, _ = filter_fresh_news(raw, now_utc=NOW)
        self.assertEqual(len(items), 1)


class DeduplicationTests(unittest.TestCase):
    def test_identical_headlines_with_different_urls_are_deduplicated(self):
        rows, _ = filter_fresh_news([
            entry(title="Nifty rises after bank rally - Source A", published=NOW - timedelta(minutes=10), link="https://a.example/a"),
            entry(title="Nifty rises after bank rally - Source B", published=NOW - timedelta(minutes=5), link="https://b.example/b"),
        ], now_utc=NOW)
        result = deduplicate_headlines(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "Source B")

    def test_case_whitespace_and_punctuation_are_normalized(self):
        self.assertEqual(normalize_headline("  NIFTY: rises, after  BANK rally! "),
                         normalize_headline("nifty rises after bank rally"))
        rows, _ = filter_fresh_news([
            entry(title="NIFTY rises after BANK rally! - A", published=NOW - timedelta(minutes=10)),
            entry(title="nifty rises after bank rally - B", published=NOW - timedelta(minutes=5), link="https://other.example/story"),
        ], now_utc=NOW)
        self.assertEqual(len(deduplicate_headlines(rows)), 1)

    def test_obvious_near_duplicate_syndication_is_deduplicated(self):
        rows, _ = filter_fresh_news([
            entry(title="Indian markets fall sharply as banking stocks slide after global selloff - A", published=NOW - timedelta(minutes=10)),
            entry(title="Indian markets fall sharply as banking stocks slide after global selloff today - B", published=NOW - timedelta(minutes=5), link="https://other.example/story"),
        ], now_utc=NOW)
        self.assertEqual(len(deduplicate_headlines(rows)), 1)

    def test_unrelated_headlines_are_not_merged(self):
        rows, _ = filter_fresh_news([
            entry(title="Nifty rises after bank rally - A", published=NOW - timedelta(minutes=10)),
            entry(title="RBI holds policy rates steady as inflation moderates - B", published=NOW - timedelta(minutes=5), link="https://other.example/story"),
        ], now_utc=NOW)
        self.assertEqual(len(deduplicate_headlines(rows)), 2)

    def test_newer_update_does_not_make_old_original_copy_win(self):
        old = entry(title="Same long syndicated headline about banking stocks and market direction - A",
                    published=NOW - timedelta(days=10), updated=NOW - timedelta(minutes=1), link="https://a.example/story")
        fresh = entry(title="Same long syndicated headline about banking stocks and market direction - B",
                      published=NOW - timedelta(minutes=8), updated=NOW - timedelta(minutes=1), link="https://b.example/story")
        rows, _ = filter_fresh_news([old, fresh], now_utc=NOW, allow_updated_fallback=True)
        result = deduplicate_headlines(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["link"], "https://b.example/story")


class SnapshotAndGeminiIsolationTests(unittest.TestCase):
    def test_only_verified_items_appear_in_snapshot_and_gemini_context(self):
        raw = {
            "entries": [
                {"entry": entry(title="Fresh Nifty headline - Daily Source", published=NOW - timedelta(minutes=10)), "feed_url": "https://news.google.com/rss"},
                {"entry": entry(title="Old Nifty headline - Old Source", published=NOW - timedelta(days=14), updated=NOW - timedelta(minutes=44)), "feed_url": "https://news.google.com/rss"},
                {"entry": entry(title="Missing timestamp headline - Unknown", published=None), "feed_url": "https://news.google.com/rss"},
            ], "feeds_ok": 1, "feeds_failed": 0, "errors": [],
        }
        snapshot = build_fresh_news_snapshot(raw, now_utc=NOW)
        context = build_gemini_news_context(snapshot, now_utc=NOW)
        self.assertEqual(snapshot["news_status"], "VALID")
        self.assertEqual(snapshot["verified_count"], 1)
        self.assertIn("Fresh Nifty headline", context)
        self.assertNotIn("Old Nifty headline", context)
        self.assertNotIn("Missing timestamp headline", context)
        self.assertIn(snapshot["snapshot_id"], context)

    def test_expired_snapshot_and_classifications_are_rejected(self):
        snapshot = build_fresh_news_snapshot(
            [entry(title="Fresh Nifty headline - Daily Source", published=NOW - timedelta(minutes=10))], now_utc=NOW
        )
        later = NOW + timedelta(seconds=61)
        valid, reason = validate_news_snapshot(snapshot, now_utc=later)
        self.assertFalse(valid)
        self.assertEqual(reason, "snapshot expired")
        self.assertEqual(verified_items_for_gemini(snapshot, now_utc=later), [])
        self.assertIn("NEWS_UNAVAILABLE", build_gemini_news_context(snapshot, now_utc=later))

    def test_news_that_ages_out_is_removed_when_raw_feed_is_revalidated(self):
        raw = [entry(title="Aged Nifty headline - Source", published=NOW - timedelta(hours=17, minutes=59))]
        first = build_fresh_news_snapshot(raw, now_utc=NOW)
        later = build_fresh_news_snapshot(raw, now_utc=NOW + timedelta(minutes=2))
        self.assertEqual(first["verified_count"], 1)
        self.assertEqual(later["verified_count"], 0)
        self.assertEqual(later["news_status"], "STALE")

    def test_empty_or_failed_feed_is_explicitly_unavailable(self):
        empty = build_fresh_news_snapshot({"entries": [], "feeds_ok": 0, "feeds_failed": 3, "errors": ["timeout"]}, now_utc=NOW)
        context = build_gemini_news_context(empty, now_utc=NOW)
        self.assertEqual(empty["news_status"], "UNAVAILABLE")
        self.assertIn("NEWS_UNAVAILABLE", context)
        self.assertIn("not evidence", context)

    def test_feed_fetch_success_uses_injected_parser_and_keeps_source_provenance(self):
        class Response:
            status_code = 200
            content = b"<rss/>"
        class Feed:
            entries = [entry(title="Fresh feed article - Publisher", published=NOW - timedelta(minutes=3))]
            bozo = 0
        result = fetch_raw_news(
            ["https://news.google.com/rss"],
            request_get=lambda *a, **k: Response(),
            parser=lambda _content: Feed(),
        )
        self.assertEqual(result["feeds_ok"], 1)
        self.assertEqual(result["feeds_failed"], 0)
        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(result["entries"][0]["feed_url"], "https://news.google.com/rss")

    def test_invalid_article_url_is_rejected(self):
        items, stats = filter_fresh_news(
            [entry(published=NOW - timedelta(minutes=5), link="javascript:alert(1)")], now_utc=NOW
        )
        self.assertEqual(items, [])
        self.assertEqual(stats["rejected_invalid_url_count"], 1)

    def test_feed_fetch_failure_does_not_require_network(self):
        class Response:
            status_code = 503
            content = b""
        result = fetch_raw_news(["https://example.com/rss"], request_get=lambda *a, **k: Response(), parser=lambda _: {})
        self.assertEqual(result["feeds_failed"], 1)
        self.assertEqual(result["entries"], [])
        snapshot = build_fresh_news_snapshot(result, now_utc=NOW)
        self.assertEqual(snapshot["news_status"], "UNAVAILABLE")


    def test_cached_classification_cannot_reintroduce_rejected_article(self):
        raw = [entry(title="Still-fresh Nifty headline - Source", published=NOW - timedelta(minutes=10))]
        snapshot = build_fresh_news_snapshot(raw, now_utc=NOW)
        item = snapshot["items"][0]
        cached = {
            item["title"]: {
                "sentiment": "BULLISH", "impact": "HIGH",
                "headline_hash": item["headline_hash"], "published_at_utc": item["published_at_utc"],
            },
            "Expired title - Source": {
                "sentiment": "BEARISH", "impact": "HIGH",
                "headline_hash": "oldhash", "published_at_utc": (NOW - timedelta(days=14)).isoformat(),
            },
        }
        retained = prune_classifications(cached, snapshot["items"])
        self.assertEqual(set(retained), {item["title"]})
        self.assertEqual(retained[item["title"]]["sentiment"], "BULLISH")

    def test_expired_pending_alert_is_removed_before_dispatch(self):
        snapshot = build_fresh_news_snapshot(
            [entry(title="One current headline - Source", published=NOW - timedelta(minutes=10))], now_utc=NOW
        )
        current_title = snapshot["items"][0]["title"]
        pending = {current_title, "Expired old headline - Source"}
        retained = prune_pending_alerts(pending, snapshot["items"])
        self.assertEqual(retained, {current_title})

    def test_deduplication_occurs_before_final_headline_limit(self):
        raw = [
            entry(title="First market headline - A", published=NOW - timedelta(minutes=10)),
            entry(title="First market headline - B", published=NOW - timedelta(minutes=9), link="https://other.example/a"),
            entry(title="Second market headline - C", published=NOW - timedelta(minutes=8), link="https://other.example/b"),
        ]
        snapshot = build_fresh_news_snapshot(raw, now_utc=NOW, max_items=2)
        self.assertEqual(snapshot["verified_count"], 2)
        self.assertEqual(snapshot["deduplicated_count"], 1)
        self.assertEqual(len({item["headline_hash"] for item in snapshot["items"]}), 2)

    def test_stale_candidates_are_not_allowed_into_a_fresh_snapshot_prompt(self):
        fresh_raw = entry(title="Verified current market story - Source", published=NOW - timedelta(minutes=10))
        stale_raw = entry(title="Old story must not reach Gemini - Source", published=NOW - timedelta(days=14))
        snapshot = build_fresh_news_snapshot([fresh_raw], now_utc=NOW)
        fresh_item = snapshot["items"][0]
        # Simulate a stale cached item offered by an old UI/classification path.
        stale_items, _ = filter_fresh_news([stale_raw], now_utc=NOW)
        self.assertEqual(stale_items, [])
        context = build_gemini_news_context(snapshot, now_utc=NOW)
        self.assertIn(fresh_item["headline"], context)
        self.assertNotIn("Old story must not reach Gemini", context)

    def test_raw_feed_can_be_revalidated_as_time_advances_without_feed_cache(self):
        raw = [entry(title="A story near expiry - Source", published=NOW - timedelta(hours=17, minutes=59))]
        first = build_fresh_news_snapshot(raw, now_utc=NOW)
        later = build_fresh_news_snapshot(raw, now_utc=NOW + timedelta(minutes=2))
        self.assertEqual(first["verified_count"], 1)
        self.assertEqual(later["verified_count"], 0)

    def test_same_snapshot_is_usable_by_ui_and_verdict_helper(self):
        snapshot = build_fresh_news_snapshot(
            [entry(title="Consistent Nifty headline - Daily Source", published=NOW - timedelta(minutes=10))], now_utc=NOW
        )
        self.assertEqual(snapshot["items"], verified_items_for_gemini(snapshot, now_utc=NOW))
        self.assertIn("Consistent Nifty headline", build_gemini_news_context(snapshot, now_utc=NOW))


class GoogleNewsUrlTests(unittest.TestCase):
    def test_recency_operator_is_inside_the_search_query(self):
        url = google_news_rss_url("Indian Stock Market Nifty Sensex", 18)
        self.assertEqual(
            url,
            "https://news.google.com/rss/search?q=Indian+Stock+Market+Nifty+Sensex+when:18h&hl=en-IN&gl=IN&ceid=IN:en",
        )

    def test_default_window_follows_the_python_freshness_constant(self):
        from services.news_service import MAX_NEWS_AGE_HOURS
        self.assertIn(f"+when:{MAX_NEWS_AGE_HOURS}h&", google_news_rss_url("Nifty"))

    def test_window_tokens(self):
        self.assertEqual(google_news_when_window(1), "1h")
        self.assertEqual(google_news_when_window(18), "18h")
        self.assertEqual(google_news_when_window(18.2), "19h")   # rounded UP so Google never drops an item we would accept
        self.assertEqual(google_news_when_window(72), "72h")
        self.assertEqual(google_news_when_window(73), "4d")      # long windows switch to whole days
        self.assertEqual(google_news_when_window(24 * 7), "7d")

    def test_invalid_windows_raise(self):
        for bad in (0, -5, None):
            with self.assertRaises(ValueError):
                google_news_when_window(bad)

    def test_query_is_normalised_and_encoded(self):
        url = google_news_rss_url("  RBI   Fed & FII ", 12)
        self.assertIn("q=RBI+Fed+%26+FII+when:12h&", url)        # whitespace collapsed, ampersand encoded
        self.assertEqual(url.count("when:"), 1)

    def test_every_feed_url_carries_exactly_one_recency_operator(self):
        queries = ["Indian Stock Market Nifty Sensex", "Global Market Wall Street Crude Oil", "RBI Fed inflation FII rupee"]
        for q in queries:
            url = google_news_rss_url(q, 18)
            self.assertEqual(url.count("when:18h"), 1)
            self.assertNotIn("when:1d", url)
            self.assertTrue(url.endswith("hl=en-IN&gl=IN&ceid=IN:en"))


if __name__ == "__main__":
    unittest.main()
