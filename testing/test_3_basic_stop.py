from playwright.sync_api import Playwright, sync_playwright, expect, Page

import yaml
import time

def load_config(filename="./testing/config.yaml"):
    with open(filename, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return config

# Load configuration
configuration = load_config()

if configuration:
    instance_location = configuration.get("instance_location")
    username = configuration.get("username")
    password = configuration.get("password")
    headless = configuration.get("headless") == "True"

    # Print out the loaded values (for demonstration)
    print("Instance Location:", instance_location)
    print("Username:", username)
    print("Password:", password)
    print("Headless:", headless)

else:
    print("Error in loading the configuration.")

def test_stop_container(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=headless)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Navigate to Splunk App for Data Science and Deep Learning app").click()
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Containers", exact=True).click()
    expect(page.get_by_text("RUNNING",exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_role("button", name="STOP")).to_be_enabled(timeout=60000)
    time.sleep(1.0)
    page.get_by_role("button", name="STOP").click()
    expect(page.get_by_text("NOT RUNNING")).to_be_visible(timeout=60000)
    # ---------------------
    context.close()
    browser.close()