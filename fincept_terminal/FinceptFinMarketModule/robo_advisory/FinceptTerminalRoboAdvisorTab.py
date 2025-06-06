from textual.containers import VerticalScroll
from textual.app import ComposeResult
from textual.widgets import Collapsible, Static, OptionList, Button, DataTable
import requests
from urllib import parse
import pandas as pd
import yfinance as yf
import asyncio

class RoboAdvisorTab(VerticalScroll):
    """Robo-Advisor Interface"""

    def compose(self) -> ComposeResult:
        """Compose the layout of the Robo-Advisor Tab."""
        yield Static("Robo-Advisor", classes="header")
        yield Static("Create Your Portfolio Based on AI and Data-Driven Insights", classes="description")

        # Collapsible for Country Selection
        with Collapsible(title="Select a Country", collapsed_symbol=">"):
            yield OptionList(id="country_selector")

        # Collapsible for Sector Selection
        with Collapsible(title="Select a Sector", collapsed_symbol=">"):
            yield OptionList(id="sector_selector")

        # Collapsible for Industry Selection
        with Collapsible(title="Select an Industry", collapsed_symbol=">"):
            yield OptionList(id="industry_selector")

        yield Static("Portfolio Analysis", id="portfolio_analysis")
        yield Button("Run Analysis", id="run_analysis_button")
        yield DataTable(id="generated_portfolio")

    async def on_mount(self):
        """Triggered when the component is mounted. Notify the user."""
        self.app.notify("Loading country options...")
        await asyncio.to_thread(self.populate_country_list)


    def populate_country_list(self):
        """Populate the OptionList with available countries."""
        try:
            continent_countries = {
                "Asia": [
                    "Afghanistan", "Bangladesh", "Cambodia", "China", "India", "Indonesia", "Japan", "Kazakhstan",
                    "Kyrgyzstan", "Malaysia", "Mongolia", "Myanmar", "Philippines", "Singapore", "South Korea",
                    "Taiwan",
                    "Thailand", "Vietnam"
                ],
                "Europe": [
                    "Austria", "Azerbaijan", "Belgium", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland",
                    "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania",
                    "Luxembourg", "Malta", "Monaco", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Russia",
                    "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "United Kingdom"
                ],
                "Africa": [
                    "Botswana", "Egypt", "Gabon", "Ghana", "Ivory Coast", "Kenya", "Morocco", "Mozambique", "Namibia",
                    "Nigeria", "South Africa", "Zambia"
                ],
                "North America": [
                    "Bahamas", "Barbados", "Belize", "Bermuda", "Canada", "Cayman Islands", "Costa Rica", "Mexico",
                    "Panama", "United States"
                ],
                "South America": [
                    "Argentina", "Brazil", "Chile", "Colombia", "Peru", "Uruguay"
                ],
                "Oceania": [
                    "Australia", "Fiji", "New Zealand", "Papua New Guinea"
                ],
                "Middle East": [
                    "Israel", "Jordan", "Saudi Arabia", "United Arab Emirates", "Qatar"
                ]
            }
            country_list = self.query_one("#country_selector", OptionList)
            countries = [country for sublist in continent_countries.values() for country in sublist]

            for country in countries:
                country_list.add_option(country)
            self.app.notify("Country list loaded successfully!")
        except Exception as e:
            self.app.notify(f"Error loading countries: {e}", severity="error")

    async def populate_sector_list(self, country):
        """Populate the OptionList with sectors for the selected country."""
        try:
            sectors = await asyncio.to_thread(self.fetch_sectors_by_country, country)
            sector_list = self.query_one("#sector_selector", OptionList)
            sector_list.clear_options()
            for sector in sectors:
                sector_list.add_option(sector)
            self.app.notify(f"Sectors for {country} loaded successfully!")
        except Exception as e:
            self.app.notify(f"Error loading sectors: {e}", severity="error")

    async def populate_industry_list(self, country, sector):
        """Populate the OptionList with industries for the selected sector in the given country."""
        try:
            industries = await asyncio.to_thread(self.fetch_industries_by_sector, country, sector)
            industry_list = self.query_one("#industry_selector", OptionList)
            industry_list.clear_options()
            for industry in industries:
                industry_list.add_option(industry)
            self.app.notify(f"Industries for {sector} in {country} loaded successfully!")
        except Exception as e:
            self.app.notify(f"Error loading industries: {e}", severity="error")

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        """Handle option selection events."""
        if event.option_list.id == "country_selector":
            selected_country = event.option_list.get_option_at_index(event.option_index).prompt
            self.selected_country = selected_country
            self.app.notify(f"Selected Country: {self.selected_country}")
            await asyncio.create_task(self.populate_sector_list(self.selected_country))

        elif event.option_list.id == "sector_selector":
            selected_sector = event.option_list.get_option_at_index(event.option_index).prompt
            self.selected_sector = selected_sector
            self.app.notify(f"Selected Sector: {self.selected_sector}")
            await asyncio.create_task(self.populate_industry_list(self.selected_country, self.selected_sector))

        elif event.option_list.id == "industry_selector":
            selected_industry = event.option_list.get_option_at_index(event.option_index).prompt
            self.selected_industry = selected_industry
            self.app.notify(f"Selected Industry: {self.selected_industry}")

    @staticmethod
    def fetch_sectors_by_country(country):
        url = f"https://fincept.share.zrok.io/FinanceDB/equities/sectors_and_industries_and_stocks?filter_column=country&filter_value={country}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("sectors", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error fetching sectors for {country}: {e}")

    @staticmethod
    def fetch_industries_by_sector(country, sector):
        sector_encoded = parse.quote(sector)
        url = f"https://fincept.share.zrok.io/FinanceDB/equities/sectors_and_industries_and_stocks?filter_column=country&filter_value={country}&sector={sector_encoded}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("industries", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error fetching industries for {sector} in {country}: {e}")


    def fetch_stocks_by_industry(self, country, sector, industry):
        """
        Fetch available stocks for the selected industry in the given sector and country using the requests library.
        """
        try:
            # URL encode the parameters
            sector_encoded = parse.quote(sector)
            industry_encoded = parse.quote(industry)
            url = (
                f"https://fincept.share.zrok.io/FinanceDB/equities/sectors_and_industries_and_stocks"
                f"?filter_column=country&filter_value={country}&sector={sector_encoded}&industry={industry_encoded}"
            )

            # Make the GET request
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad HTTP status codes

            # Parse the JSON response
            stock_data = response.json()
            self.app.notify("stock data found")
            # Convert response to a DataFrame and return
            return pd.DataFrame(stock_data)
        except requests.exceptions.RequestException as e:
            self.app.notify("no stock data found")
            return pd.DataFrame()  # Return an empty DataFrame on failure


    async def on_button_pressed(self, event):
        """Handle button press events."""
        if event.button.id == "run_analysis_button":
            self.notify("Analyzing Portfolio...")
            await asyncio.create_task(self.run_robo_analysis())

    async def run_robo_analysis(self):
        """Run the full Robo-Advisor analysis."""
        self.app.notify("Running Robo-Advisor Analysis...")

        # Ensure selections are made
        if not hasattr(self, "selected_country") or not hasattr(self, "selected_sector") or not hasattr(self, "selected_industry"):
            self.app.notify("Please select a country, sector, and industry before running analysis.", severity="error")
            return

        country = self.selected_country
        sector = self.selected_sector
        industry = self.selected_industry

        # Fetch stocks based on the selected options
        stocks = await asyncio.to_thread(self.fetch_stocks_by_industry, country, sector, industry)
        if stocks.empty:
            self.app.notify(f"No stocks found for {industry} in {sector} ({country}).", severity="warning")
            return

        # Perform Technical & Fundamental Analysis
        portfolio = await asyncio.create_task(self.run_technical_fundamental_analysis(stocks))

        if portfolio.empty:
            self.app.notify("No suitable stocks found after analysis.", severity="warning")
            return

        # ✅ Update DataTable instead of Static
        data_table = self.query_one("#generated_portfolio", DataTable)
        data_table.clear()

        # ✅ Add Columns if not already added
        columns = list(portfolio.columns)
        for col in columns:
            data_table.add_column(col)

        # ✅ Add Data to DataTable
        for row in portfolio.itertuples(index=False):
            data_table.add_row(*row)

        self.app.notify("Portfolio generated successfully!")

    async def run_technical_fundamental_analysis(self, stocks):
        """Perform the actual technical and fundamental analysis (runs in a background thread)."""
        stock_analysis = []

        # Validate input stocks DataFrame
        if stocks.empty or "symbol" not in stocks.columns:
            self.notify("No stocks available for analysis.", severity="warning")
            return pd.DataFrame()

        for symbol in stocks["symbol"]:
            try:
                stock = yf.Ticker(symbol)
                info = await asyncio.to_thread(lambda: stock.info)  # Fetch stock info safely in a background thread
                hist = await asyncio.to_thread(lambda: stock.history(period="1y"))  # Fetch historical data safely

                # Validate historical data
                if hist.empty or "Close" not in hist.columns:
                    self.notify(f"{symbol}: No valid historical data. Skipping.", severity="information")
                    continue

                # Calculate SMAs
                sma_50 = hist["Close"].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
                sma_200 = hist["Close"].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
                current_price = hist["Close"].iloc[-1]

                # Skip if any SMA is unavailable
                if sma_50 is None or sma_200 is None:
                    self.notify(f"{symbol}: Not enough data to calculate SMAs. Skipping.", severity="information")
                    continue

                # Fetch market cap safely
                market_cap = info.get("marketCap", None)
                if market_cap is None:
                    self.notify(f"{symbol}: No market cap data available. Skipping.", severity="information")
                    continue

                # Scoring logic
                fundamental_score = 10 if market_cap > 1e10 else (7 if market_cap > 1e9 else 5)
                technical_score = 10 if sma_50 > sma_200 else 5
                combined_score = fundamental_score + technical_score

                # Append stock analysis data
                stock_analysis.append({
                    "symbol": symbol,
                    "name": info.get("longName", symbol),
                    "market_cap": market_cap,
                    "current_price": current_price,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "fundamental_score": fundamental_score,
                    "technical_score": technical_score,
                    "combined_score": combined_score,
                })

            except Exception as e:
                self.notify(f"Error analyzing {symbol}: {e}", severity="error")

        # Convert analysis results to DataFrame
        df = pd.DataFrame(stock_analysis)

        # Handle empty DataFrame scenario
        if df.empty:
            self.notify("No valid stock data after analysis.", severity="warning")
            return pd.DataFrame()

        # Ensure 'combined_score' exists before sorting
        if "combined_score" in df.columns:
            return df.sort_values("combined_score", ascending=False).head(10)
        else:
            self.notify("No valid combined_score column available.", severity="error")
            return pd.DataFrame()
