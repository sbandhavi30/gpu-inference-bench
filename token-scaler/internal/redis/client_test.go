package redis_test

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
	"token-scaler/internal/redis"
)

func newTestClient(t *testing.T) (*redis.Client, *miniredis.Miniredis) {
	t.Helper()
	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("miniredis start: %v", err)
	}
	t.Cleanup(mr.Close)
	c, err := redis.NewClient(mr.Addr(), "")
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { c.Close() })
	return c, mr
}

func TestLoadScripts_ReturnsSHAs(t *testing.T) {
	c, _ := newTestClient(t)
	sm, err := c.LoadScripts(context.Background())
	if err != nil {
		t.Fatalf("LoadScripts: %v", err)
	}
	if sm.PublisherSHA == "" || sm.EvaluatorSHA == "" {
		t.Fatal("expected non-empty SHAs")
	}
}

func TestGetHashLength_Empty(t *testing.T) {
	c, _ := newTestClient(t)
	n, err := c.GetHashLength(context.Background(), "token_backlog:q1")
	if err != nil {
		t.Fatalf("GetHashLength: %v", err)
	}
	if n != 0 {
		t.Fatalf("want 0, got %d", n)
	}
}

func TestGetHashLength_NonEmpty(t *testing.T) {
	c, mr := newTestClient(t)
	mr.HSet("token_backlog:q1", "job1", "1000")
	n, err := c.GetHashLength(context.Background(), "token_backlog:q1")
	if err != nil {
		t.Fatalf("GetHashLength: %v", err)
	}
	if n != 1 {
		t.Fatalf("want 1, got %d", n)
	}
}

func TestEvaluator_EmptyBacklog(t *testing.T) {
	c, _ := newTestClient(t)
	if _, err := c.LoadScripts(context.Background()); err != nil {
		t.Fatal(err)
	}
	total, err := c.ExecuteEvaluator(context.Background(), "q1",
		time.Now().Unix(), "86400", "100")
	if err != nil {
		t.Fatalf("ExecuteEvaluator: %v", err)
	}
	if total != 0 {
		t.Fatalf("want 0, got %d", total)
	}
}

func TestEvaluator_SumsTokens(t *testing.T) {
	c, mr := newTestClient(t)
	if _, err := c.LoadScripts(context.Background()); err != nil {
		t.Fatal(err)
	}
	mr.HSet("token_backlog:q1", "job1", "1000", "job2", "2500", "job3", "500")
	future := time.Now().Add(1 * time.Hour).Unix()
	mr.ZAdd("token_expiry:q1", float64(future), "job1")
	mr.ZAdd("token_expiry:q1", float64(future), "job2")
	mr.ZAdd("token_expiry:q1", float64(future), "job3")

	total, err := c.ExecuteEvaluator(context.Background(), "q1",
		time.Now().Unix(), "86400", "100")
	if err != nil {
		t.Fatalf("ExecuteEvaluator: %v", err)
	}
	if total != 4000 {
		t.Fatalf("want 4000, got %d", total)
	}
}

func TestEvaluator_PurgesExpiredJobs(t *testing.T) {
	c, mr := newTestClient(t)
	if _, err := c.LoadScripts(context.Background()); err != nil {
		t.Fatal(err)
	}
	past := time.Now().Add(-1 * time.Hour).Unix()
	future := time.Now().Add(1 * time.Hour).Unix()
	mr.HSet("token_backlog:q1", "expired", "9999", "active", "1000")
	mr.ZAdd("token_expiry:q1", float64(past), "expired")
	mr.ZAdd("token_expiry:q1", float64(future), "active")

	total, err := c.ExecuteEvaluator(context.Background(), "q1",
		time.Now().Unix(), "86400", "100")
	if err != nil {
		t.Fatalf("ExecuteEvaluator: %v", err)
	}
	if total != 1000 {
		t.Fatalf("want 1000, got %d", total)
	}
	// Verify expired job removed from hash
	rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
	defer rdb.Close()
	exists, _ := rdb.HExists(context.Background(), "token_backlog:q1", "expired").Result()
	if exists {
		t.Fatal("expired job should have been purged from hash")
	}
}

func TestEvaluator_ChunkCeiling_100PerCycle(t *testing.T) {
	c, mr := newTestClient(t)
	if _, err := c.LoadScripts(context.Background()); err != nil {
		t.Fatal(err)
	}
	past := time.Now().Add(-1 * time.Hour).Unix()
	// Insert 105 expired jobs
	rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
	defer rdb.Close()
	for i := 0; i < 105; i++ {
		id := "job" + strconv.Itoa(i)
		mr.HSet("token_backlog:q1", id, "100")
		mr.ZAdd("token_expiry:q1", float64(past), id)
	}
	// Cycle 1: only 100 purged
	total1, err := c.ExecuteEvaluator(context.Background(), "q1",
		time.Now().Unix(), "86400", "100")
	if err != nil {
		t.Fatalf("cycle1: %v", err)
	}
	remaining1, err := rdb.HLen(context.Background(), "token_backlog:q1").Result()
	if err != nil {
		t.Fatalf("cycle1 hlen: %v", err)
	}
	if remaining1 != 5 {
		t.Fatalf("cycle1: want 5 remaining, got %d", remaining1)
	}
	// total1 = sum of remaining 5 * 100 = 500
	if total1 != 500 {
		t.Fatalf("cycle1: want sum 500, got %d", total1)
	}

	// Cycle 2: remaining 5 purged
	total2, err := c.ExecuteEvaluator(context.Background(), "q1",
		time.Now().Unix(), "86400", "100")
	if err != nil {
		t.Fatalf("cycle2: %v", err)
	}
	if total2 != 0 {
		t.Fatalf("cycle2: want 0, got %d", total2)
	}
	remaining2, err := rdb.HLen(context.Background(), "token_backlog:q1").Result()
	if err != nil {
		t.Fatalf("cycle2 hlen: %v", err)
	}
	if remaining2 != 0 {
		t.Fatalf("cycle2: want 0 remaining, got %d", remaining2)
	}
}

func TestEvaluator_NoscriptRetry(t *testing.T) {
	c, mr := newTestClient(t)
	if _, err := c.LoadScripts(context.Background()); err != nil {
		t.Fatal(err)
	}
	// SCRIPT FLUSH clears all cached scripts, simulating a Redis restart (NOSCRIPT scenario).
	raw := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
	defer raw.Close()
	if err := raw.Do(context.Background(), "SCRIPT", "FLUSH").Err(); err != nil {
		t.Fatalf("script flush: %v", err)
	}
	// ExecuteEvaluator must detect NOSCRIPT, reload scripts, and succeed.
	total, err := c.ExecuteEvaluator(context.Background(), "q1",
		time.Now().Unix(), "86400", "100")
	if err != nil {
		t.Fatalf("expected retry to succeed after NOSCRIPT: %v", err)
	}
	if total != 0 {
		t.Fatalf("want 0, got %d", total)
	}
}
