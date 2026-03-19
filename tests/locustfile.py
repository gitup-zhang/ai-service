from locust import HttpUser, task, between

class AIServiceUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"

    @task(3)
    def test_health(self):
        """测试服务健康检查接口 (高频访问)"""
        self.client.get("/health", name="Health Check")

    @task(1)
    def test_chat_single(self):
        """测试单轮问答接口"""
        payload = {
            "question": "国家最近有什么新的助学贷款政策吗？",
            "history": []
        }
        self.client.post("/api/ai/chat/ask", json=payload, name="Chat - Single Round")
