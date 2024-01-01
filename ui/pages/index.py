from st_pages import Page, show_pages, add_page_title


class UI:
    @staticmethod
    def setup_pages(show_title=True):
        if show_title:
            add_page_title()

        # Define your pages
        pages = [
            Page("ui/pages/analyse.py", "Analyses", "🤖", is_section=True),
            Page("ui/pages/about.py", "About", "🏠"),
        ]

        show_pages(pages)
