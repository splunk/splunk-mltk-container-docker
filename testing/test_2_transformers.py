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

def test_transformers_finetune_en(playwright: Playwright,containername) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Navigate to Splunk App for Data Science and Deep Learning app").click()
    page.get_by_role("link", name="Assistants ▾").click()
    page.get_by_role("link", name="› Deep Learning NLP").click()
    page.get_by_role("link", name="Deep Learning Text Classification").click()
    page.locator("#content2").get_by_role("link").click()
    page.get_by_label("Search Button").click()
    page.get_by_label("Select...").click()
    page.get_by_role("option", name="bert_classification_en").click()
    page.get_by_label("Target model name").click()
    page.get_by_label("Target model name").fill("test")
    page.get_by_text("Select languageenSelect base modelbert_classification_enClearTarget model nameBa").click()
    page.get_by_text("Select languageenSelect base modelbert_classification_enClearTarget model nameBa").click()
    page.get_by_label("Batch Size").click()
    page.get_by_role("button", name="Review Fine-Tuning SPL").click()
    page.get_by_role("button", name="Run Fine-Tuning SPL").click()
    page.get_by_role("button", name="Done").click(timeout=300000)

    # ---------------------
    context.close()
    browser.close()
