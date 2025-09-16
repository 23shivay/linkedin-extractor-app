# app.py

import asyncio
import streamlit as st
from playwright.async_api import async_playwright
from crawl4ai import AsyncWebCrawler, LLMExtractionStrategy, LLMConfig, CrawlerRunConfig
import json
import tempfile
import os
import sys

# Your API keys should NOT be hardcoded here.
# For local testing, you can use environment variables or a .streamlit/secrets.toml file.
# For deployment, Streamlit Cloud uses its secrets management.
# Example for local testing: os.environ.get("GROQ_API_KEY")

async def get_authenticated_html(li_at_cookie: str, url: str):
    """
    Uses Playwright to get authenticated HTML from a LinkedIn activity URL.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.add_cookies([{
            'name': 'li_at',
            'value': li_at_cookie,
            'domain': '.linkedin.com',
            'path': '/'
        }])
        page = await context.new_page()
        
        st.status("🌐 Loading LinkedIn with Playwright...", expanded=True)
        await page.goto(url, wait_until="networkidle", timeout=60000)
        await asyncio.sleep(5)
        
        current_url = page.url
        if "login" in current_url or "checkpoint" in current_url:
            st.error("❌ LinkedIn redirected to login - cookie is expired or invalid!")
            await browser.close()
            return None
        
        feed_html = None
        selectors_to_try = [
            'ul.display-flex.flex-wrap.list-style-none.justify-center',
            '.scaffold-finite-scroll__content',
            '.application-outlet main',
        ]
        
        for selector in selectors_to_try:
            try:
                feed_element = await page.query_selector(selector)
                if feed_element:
                    feed_html = await feed_element.inner_html()
                    st.success(f"✅ Found feed container with selector: {selector}")
                    break
            except Exception:
                continue
        
        if not feed_html:
            st.warning("⚠️ No specific container found, using full page HTML.")
            feed_html = await page.content()
        
        await browser.close()
        return feed_html

async def extract_with_crawl4ai(html_content: str):
    """
    Uses Crawl4AI to extract structured data from the provided HTML content.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(f"<html><body>{html_content}</body></html>")
        temp_html_path = f.name
    
    try:
        file_url = f"file://{os.path.abspath(temp_html_path)}"
        
        llm_api_key = st.secrets["GROQ_API_KEY"]
        
        extraction_strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="groq/meta-llama/llama-4-scout-17b-16e-instruct",
                api_token=llm_api_key,
            ),
            instruction=(
                "You are analyzing LinkedIn activity feed HTML. Focus ONLY on the FIRST 3 posts at the TOP.\n"
                "Extract the 'apply_link', 'eligibility', 'company_name', 'stipend', 'job_title', 'location', and 'timestamp'.\n"
                "Return as a JSON array."
            ),
            extract_type="schema",
            schema="""[{
                "apply_link": "string or null",
                "eligibility": "string or null", 
                "company_name": "string or null",
                "stipend": "string or null",
                "job_title": "string or null",
                "location": "string or null",
                "timestamp": "string or null"
            }]""",
            extra_args={"temperature": 0.0, "max_tokens": 4096},
            verbose=True,
        )
        
        config = CrawlerRunConfig(extraction_strategy=extraction_strategy)
        
        st.status("🤖 Running Crawl4AI extraction...", expanded=True)
        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun(url=file_url, config=config)
            
            for result in results:
                if result.success:
                    try:
                        data = json.loads(result.extracted_content)
                        st.success("✅ Crawl4AI extraction successful!")
                        return data
                    except json.JSONDecodeError:
                        st.error("❌ Failed to parse Crawl4AI JSON output.")
                        st.json({"raw_content": result.extracted_content})
                        return []
                else:
                    st.error("❌ Crawl4AI extraction failed.")
                    return []
                    
    finally:
        try:
            os.unlink(temp_html_path)
        except OSError:
            pass

async def main_app():
    """Main Streamlit application logic."""
    st.title("LinkedIn Post Extractor")
    st.markdown("Enter your LinkedIn 'li_at' cookie and a profile URL to extract recent posts.")

    # Input fields
    li_at_cookie = st.text_input("Enter your 'li_at' cookie:", type="password")
    activity_url = st.text_input("Enter LinkedIn activity URL:", "https://www.linkedin.com/in/krishan-kumar08/recent-activity/all/")

    if st.button("Start Extraction"):
        if not li_at_cookie:
            st.error("Please enter your 'li_at' cookie.")
            return

        with st.status("Initializing...", expanded=True) as status:
            try:
                html_content = await get_authenticated_html(li_at_cookie, activity_url)
                if html_content:
                    extracted_data = await extract_with_crawl4ai(html_content)
                    
                    status.update(label="Extraction Complete!", state="complete", expanded=False)

                    if extracted_data:
                        st.header("📊 Final Results")
                        st.json(extracted_data, expanded=False)
                        
                        st.header("📋 Formatted Output")
                        for i, post in enumerate(extracted_data):
                            st.subheader(f"Post {i+1}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Company:** {post.get('company_name', 'N/A')}")
                                st.write(f"**Job Title:** {post.get('job_title', 'N/A')}")
                                st.write(f"**Location:** {post.get('location', 'N/A')}")
                                st.write(f"**Timestamp:** {post.get('timestamp', 'N/A')}")
                            with col2:
                                st.write(f"**Eligibility:** {post.get('eligibility', 'N/A')}")
                                st.write(f"**Stipend:** {post.get('stipend', 'N/A')}")
                                st.write(f"**Apply Link:** {post.get('apply_link', 'N/A')}")
                    else:
                        st.error("❌ No data was extracted. Check the logs above for details.")
            except Exception as e:
                st.exception(e)
                status.update(label="An error occurred", state="error", expanded=True)

# Main entry point with event loop fix
if __name__ == "__main__":
    if sys.platform == "win32":
        # On Windows, set the event loop policy to a compatible one
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_app())