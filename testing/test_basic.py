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

    # Print out the loaded values (for demonstration)
    print("Instance Location:", instance_location)
    print("Username:", username)
    print("Password:", password)

else:
    print("Error in loading the configuration.")

def test_docker_configure(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Splunk App for Data Science and Deep Learning").click()
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Setup", exact=True).click()
    page.locator("[data-test=\"button\"]").click()
    page.get_by_label("Docker Host").get_by_label("value").fill("unix://var/run/docker.sock")
    page.get_by_label("Endpoint URL", exact=True).get_by_label("value").fill("localhost")
    page.get_by_label("External URL").get_by_label("value").fill("dsdl.splunkyourface.com")
    page.get_by_label("Disabled").click()
    page.get_by_label("Jupyter Password").get_by_label("value").fill("testpassword")
    page.get_by_role("button", name="Test & Save").click()
    expect(page.get_by_text("Successfully established connection to container environment.")).to_be_visible()
    page.get_by_role("button", name="OK", exact=True).click()
    # ---------------------
    context.close()
    browser.close()

def test_start_container(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Splunk App for Data Science and Deep Learning").click()
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Containers", exact=True).click()
    page.get_by_label("Container Image").click()
    page.get_by_role("option", name=containername, exact=True).click()
    expect(page.get_by_text("NOT RUNNING",exact=True)).to_be_visible(timeout=5000)
    page.get_by_role("button", name="START").click()
    expect(page.get_by_text("RUNNING",exact=True)).to_be_visible(timeout=50000)
    # ---------------------
    context.close()
    browser.close()

def test_check_jupyter(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    time.sleep(10)
    page.goto("https://dsdl.splunkyourface.com:8888/login?next=%2Flab%3F")
    page.locator("html").click()
    page.get_by_label("Password:").click()
    page.get_by_label("Password:").fill("testpassword")
    page.get_by_label("Password:").press("Enter")
    page.goto("https://dsdl.splunkyourface.com:8888/lab")
    page.get_by_text("notebooks", exact=True).click()
    page.get_by_role("tab", name="Running Terminals and Kernels").get_by_role("img").click()
    page.get_by_role("tab", name="File Browser (⇧ ⌘ F)").locator("path").click()
    page.get_by_title("/srv").get_by_role("img").click()
    page.get_by_text("notebooks", exact=True).dblclick()
    expect(page.get_by_label("File Browser Section").get_by_text("barebone_template.ipynb")).to_be_visible(timeout=5000)
    # ---------------------
    context.close()
    browser.close()

def test_barebones_model(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    # ---------------------
    context.close()
    browser.close()


def test_stop_container(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Splunk App for Data Science and Deep Learning").click()
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Containers", exact=True).click()
    page.get_by_label("Container Image").click()
    page.get_by_role("option", name="golden-cpu", exact=True).click()
    expect(page.get_by_text("RUNNING",exact=True)).to_be_visible(timeout=5000)
    page.get_by_role("button", name="STOP").click()
    expect(page.get_by_text("NOT RUNNING")).to_be_visible(timeout=50000)
    # ---------------------
    context.close()
    browser.close()
