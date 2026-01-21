from locust import HttpUser, task

class MyUser(HttpUser):
    """My user class that defines the tasks to be performed by the user."""

    @task
    def index(self):
        """GET / - user visiting the root URL of the FastAPI app"""
        self.client.get("/")

    @task(1)
    def user(self):
        """GET /user/16 - user visiting a specific user page"""
        self.client.get("/user/16")
