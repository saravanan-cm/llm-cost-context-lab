def test_usage_for_new_user(client):
    response = client.get("/api/v1/usage")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "dev-user",
        "total_credits": "1000",
        "credits_used": "0",
        "credits_remaining": "1000",
        "request_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_cost": "0",
        "currency": "USD",
    }


def test_usage_aggregates_requests(client, provider):
    client.post("/api/v1/chat", json={"conversation_id": "c1", "message": "one"})
    provider.input_tokens, provider.output_tokens = 1000, 500
    client.post("/api/v1/chat", json={"conversation_id": "c2", "message": "two"})

    body = client.get("/api/v1/usage").json()

    assert body["request_count"] == 2
    assert body["total_input_tokens"] == 1120
    assert body["total_output_tokens"] == 680
    assert body["total_tokens"] == 1800
    # 0.000336 + (1000 * 0.40 + 500 * 1.60) / 1e6 = 0.000336 + 0.0012
    assert body["total_cost"] == "0.001536"
    assert body["credits_used"] == "1.536"
    assert body["credits_remaining"] == "998.464"
