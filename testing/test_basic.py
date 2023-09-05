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
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Splunk App for Data Science and Deep Learning",exact=True)
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Setup", exact=True).click()
    page.locator("[data-test=\"button\"]").click()
    page.get_by_label("Docker Host").get_by_label("value").fill("unix://var/run/docker.sock")
    page.get_by_label("Endpoint URL", exact=True).get_by_label("value").fill("localhost")
    page.get_by_label("External URL").get_by_label("value").fill("dsdl.splunkyourface.com")
    page.get_by_label("Disabled").click()
    page.get_by_label("Jupyter Password").get_by_label("value").fill("testpassword")
    time.sleep(1.0)
    expect(page.get_by_role("button", name="Test & Save")).to_be_enabled()
    time.sleep(1.0)
    page.get_by_role("button", name="Test & Save").click()
    time.sleep(1.0)
    expect(page.get_by_text("Successfully established connection to container environment.")).to_be_visible(timeout=60000)
    time.sleep(1.0)
    page.get_by_role("button", name="OK", exact=True).click()
    # ---------------------
    context.close()
    browser.close()

def test_start_container(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Splunk App for Data Science and Deep Learning",exact=True)
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Containers", exact=True).click()
    page.get_by_label("Container Image").click()
    page.get_by_role("option", name=containername, exact=True).click()
    expect(page.get_by_text("NOT RUNNING",exact=True)).to_be_visible(timeout=60000)
    page.get_by_role("button", name="START").click()
    expect(page.get_by_text("RUNNING",exact=True)).to_be_visible(timeout=60000)
    # ---------------------
    context.close()
    browser.close()

def test_check_jupyter(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    time.sleep(10)
    page.goto(f"https://{instance_location}:8888/login?next=%2Flab%3F")
    page.locator("html").click()
    page.get_by_label("Password:").click()
    page.get_by_label("Password:").fill("testpassword")
    page.get_by_label("Password:").press("Enter")
    page.goto(f"https://{instance_location}:8888/lab")
    page.get_by_text("notebooks", exact=True).click()
    page.get_by_role("tab", name="Running Terminals and Kernels").get_by_role("img").click()
    page.get_by_role("tab", name="File Browser (⇧ ⌘ F)").locator("path").click()
    page.get_by_title("/srv").get_by_role("img").click()
    page.get_by_text("notebooks", exact=True).dblclick()
    expect(page.get_by_label("File Browser Section").get_by_text("barebone_template.ipynb")).to_be_visible(timeout=60000)
    # ---------------------
    context.close()
    browser.close()

def test_barebones_model(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page2 = context.new_page()
    page2.goto(f"http://{instance_location}:8000/en-GB/app/search/search?q=%7C%20makeresults%20count%3D10%20%0A%7C%20streamstats%20c%20as%20i%20%0A%7C%20eval%20i%3Di-1%20%0A%7C%20fit%20MLTKContainer%20algo%3Dbarebone_template%20i%20%0A%7C%20eval%20test%3Di-predicted_index%20%0A%7C%20stats%20sum(test)%20as%20test%20%0A%7C%20eval%20test%3Dif(test%3D%3D0%2C%22SUCCESS%22%2C%22FAIL%22)&display.page.search.mode=fast&dispatch.sample_ratio=1&workload_pool=&earliest=-24h%40h&latest=now&display.page.search.tab=statistics&display.general.type=statistics")
    page2.get_by_role("link", name="Run Query Anyway").click()
    expect(page2.get_by_role("cell", name="SUCCESS", exact=True)).to_be_visible(timeout=60000)
    # ---------------------
    context.close()
    browser.close()


def test_stop_container(playwright: Playwright, containername) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto(f"http://{instance_location}:8000/en-GB/account/login?return_to=%2Fen-GB%2F")
    page.get_by_placeholder("Username").fill(username)
    page.get_by_placeholder("Password", exact=True).fill(password)
    page.get_by_placeholder("Password", exact=True).press("Enter")
    page.get_by_label("Splunk App for Data Science and Deep Learning",exact=True)
    page.get_by_role("link", name="Configuration ▾").click()
    page.get_by_role("link", name="Containers", exact=True).click()
    expect(page.get_by_text("RUNNING",exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_role("button", name="STOP")).to_be_enabled(timeout=60000)
    page.get_by_role("button", name="STOP").click()
    expect(page.get_by_text("NOT RUNNING")).to_be_visible(timeout=60000)
    # ---------------------
    context.close()
    browser.close()