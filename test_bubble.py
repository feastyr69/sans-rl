from playwright.sync_api import sync_playwright
import cv2
import numpy as np
import time

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:9222")
    page = browser.contexts[0].pages[0]
    canvas = page.locator('canvas')
    
    screenshot = canvas.screenshot(timeout=2000)
    np_img = np.frombuffer(screenshot, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    bubble_region = img[50:150, 350:500]
    white_mask = (bubble_region[:, :, 2] > 240) & (bubble_region[:, :, 1] > 240) & (bubble_region[:, :, 0] > 240)
    print("White pixels in bubble region:", np.sum(white_mask))
    
    cv2.imwrite("bubble_region.png", bubble_region)
