import os
import time
from datetime import datetime

import requests


class SplunkHEC(object):
    def __init__(self, url="", token=""):
        self.url = url + "/services/collector/event"
        self.token = token
        self.authHeader = {}

        if "splunk_hec_enabled" in os.environ:
            access_enabled = os.environ["splunk_hec_enabled"]
            if access_enabled == "1":
                self.url = os.environ["splunk_hec_url"] + "/services/collector/event"
                self.token = os.environ["splunk_hec_token"]

        if self.token:
            self.authHeader = {"Authorization": "Splunk {}".format(self.token)}

    def send_to_hec(self, jsonDict, url):
        return requests.post(url, headers=self.authHeader, json=jsonDict, verify=False)

    def send(self, events):
        return self.send_to_hec(events, self.url)

    def send_hello_world(self, num=1):
        data = []
        for i in range(0, num):
            event = {
                "event": {"message": "hello world " + str(i)},
                "time": datetime.now().timestamp(),
            }
            time.sleep(0.001)
            data.append(event)
        return self.send(data)
