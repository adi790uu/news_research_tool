import json
from llama_index.llms.ollama import Ollama
from tavily import TavilyClient
from bs4 import BeautifulSoup
import asyncio
from threading import Lock
from llama_index.core.llms import ChatMessage
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
from pydantic import BaseModel


class Query(BaseModel):
    """Data model for getting the query to search on the search engine."""

    query_to_search: str


llm = Ollama(model="llama3.2")
client = TavilyClient(api_key="enter key")


class WebDriverPool:
    def __init__(self, max_drivers=1):
        self.max_drivers = max_drivers
        self.drivers = []
        self.available_drivers = []
        self.lock = Lock()

    async def init_pool(self):
        for _ in range(self.max_drivers):
            driver = await self._create_driver()
            self.drivers.append(driver)
            self.available_drivers.append(driver)

    async def _create_driver(self) -> WebDriver:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._create_driver_sync
        )

    def _create_driver_sync(self) -> WebDriver:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--incognito")
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.page_load_strategy = "eager"
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(3)
        driver.set_page_load_timeout(10)
        return driver

    async def get_driver(self):
        while True:
            with self.lock:
                if self.available_drivers:
                    return self.available_drivers.pop()
            await asyncio.sleep(0.5)

    def release_driver(self, driver):
        with self.lock:
            self.available_drivers.append(driver)

    async def close_all(self):
        for driver in self.drivers:
            await asyncio.get_event_loop().run_in_executor(None, driver.quit)
        self.drivers.clear()
        self.available_drivers.clear()


driver_pool = WebDriverPool()


async def init_driver_pool():
    await driver_pool.init_pool()


async def close_driver_pool():
    await driver_pool.close_all()


async def get_driver():
    return await driver_pool.get_driver()


def release_driver(driver):
    driver_pool.release_driver(driver)


async def selenium_and_bs4_scraping(url: str):
    try:
        driver = await get_driver()
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        contentBody = soup.find("body").getText()
        return contentBody
    except Exception as e:
        print(e)
        return False
    finally:
        release_driver(driver)


async def main():
    await init_driver_pool()
    query = input("What is your query?")
    result = await selenium_and_bs4_scraping(query)
    # print(result)
    query0 = f"""
        clean this content up and only include relevant thing from the news.
        {result}
    """

    result1 = await llm.acomplete(query0)

    query1 = f"""
        Provide me the most relevant query from the content provided which can be put in a search engine to gather similar content from
        web.
        Here is the content: {result1}
    """

    sllm = llm.as_structured_llm(output_cls=Query)
    input_msg = ChatMessage.from_str(query1)

    output = await sllm.achat([input_msg])
    output_obj = output.raw.query_to_search
    print(output_obj)
    response = client.search(output_obj)

    similar_content = []

    for result in response["results"]:
        similar_content.append(result["content"])

    print(response["results"][0]["content"])

    prompt = f"""
        Here is the original article: {result}
        Here are similar content fetched from internet:
        {json.dumps(similar_content)}

        Tell me whether the original article is factually correct or is it a
            wrong news, give me a similarity score as well.
    """
    input_msg = ChatMessage.from_str(prompt)
    response = await llm.achat([input_msg])
    print(response)


while True:
    asyncio.run(main())
