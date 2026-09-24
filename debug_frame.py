import cv2
import numpy as np
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp('http://localhost:9222')
    page = browser.contexts[0].pages[0]
    for _ in range(5):
        # Force frame render
        page.evaluate('() => new Promise(requestAnimationFrame)')
        screenshot = page.locator('canvas').screenshot()
        img = cv2.imdecode(np.frombuffer(screenshot, dtype=np.uint8), cv2.IMREAD_COLOR)
        print("Max pixel value:", np.max(img))
