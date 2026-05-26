from site_safety_monitor.data.corpus_models import OshaSourcePage


def test_osha_source_page_requires_standard_and_url():
    page = OshaSourcePage(
        standard_number="1910.132",
        title="General requirements",
        url="https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.132",
        domain_tags=("ppe",),
    )

    assert page.standard_number == "1910.132"
    assert page.domain_tags == ("ppe",)
