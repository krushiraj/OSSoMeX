import json

import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import expect

from test_comparison_html_report import artifacts, build
from test_full_label_prediction import pipeline_factory


def test_offline_dashboard_navigation_highlights_and_mobile_layout(page, tmp_path, pipeline_factory):
    text = '🧪 We used NumPy 1.24 and ToolX 2.26. <script>alert(1)</script>'
    artifacts(tmp_path, pipeline_factory, text=text)
    appendix = tmp_path / 'appendix.json'
    appendix.write_text(json.dumps({'title': 'Capability examples', 'examples': [{
        'kind': 'illustrative_after_task_training', 'title': 'Illustration only', 'input': 'sample',
        'output': {}, 'note': 'Not a measured prediction.',
        'source_url': 'https://github.com/allenai/scicite/blob/master/README.md'}]}))
    build(tmp_path, appendix=appendix)
    requests = []
    page.on('request', lambda request: requests.append(request.url))
    page.goto((tmp_path / 'html/report.html').as_uri())
    expect(page.get_by_role('heading', name='Software extraction · comparison workbench')).to_be_visible()
    assert page.locator('script').count() == 0
    assert page.locator('.reference-panel .source-text').first.text_content() == text
    assert page.locator('.reference-panel mark.software').all_text_contents() == ['NumPy', 'ToolX']
    page.get_by_role('link', name='Passages', exact=True).click()
    expect(page.get_by_role('heading', name='Passage review')).to_be_in_viewport()
    expect(page.get_by_role('heading', name='Other field outputs (unscored)')).to_be_visible()
    raw = page.get_by_text('Raw field JSON', exact=True)
    raw.click()
    expect(raw.locator('..').locator('pre').first).to_be_visible()
    raw.click()
    for width in (1440, 390):
        page.set_viewport_size({'width': width, 'height': 900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert all(url.startswith('file:') for url in requests)
