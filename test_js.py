from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://jcw87.github.io/c2-sans-fight/")
    page.wait_for_selector("canvas")
    keys = page.evaluate("Object.keys(window)")
    print(keys)
    browser.close()
