import asyncio
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Container
from textual.widgets import Input, Button, OptionList, Static, LoadingIndicator
import json
import torch
import google.generativeai as gemai
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import default_cache_path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, HFValidationError
from fincept_terminal.FinceptSettingModule.FinceptTerminalSettingUtils import get_settings_path

SETTINGS_FILE = get_settings_path()
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment"

class YouTubeTranscriptApp(Container):
    """Textual App to search YouTube videos, fetch transcripts, and analyze sentiment."""

    # For model state tracking
    model_loaded = False
    tokenizer = None
    model = None
    download_in_progress = False

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Static("🔎 Enter Search Keyword:", id="search_label")
        yield Input(placeholder="Enter keyword...", id="search_input")
        yield Button("Search", id="search_button", variant="primary")

        yield Static("🎥 Select a Video:", id="video_label", classes="hidden")
        yield OptionList(id="video_list")

        with VerticalScroll(id="analysis_results", classes="analysis_results"):
            yield Static("Transcript", id="transcript_label", classes="hidden")
            yield Static("", id="transcript_display")

            yield Static("Sentiment Analysis", id="sentiment_label", classes="hidden")
            yield Static("", id="sentiment_display")
            yield Button("Download Sentiment Model", id="download_model_button", variant="primary", classes="hidden")
            # yield LoadingIndicator(id="download_progress", classes="hidden")
    
    def on_mount(self):
        """Check for model availability when app is mounted."""
        # Start async task to check model availability without blocking UI
        asyncio.create_task(self.check_model_availability())
    
    async def check_model_availability(self):
        """Check if the sentiment model is available locally."""
        try:
            # Try to find the model in the local cache asynchronously
            local_path = await asyncio.to_thread(
                snapshot_download, 
                repo_id=MODEL_ID, 
                local_files_only=True
            )
            
            if local_path:
                self.app.notify("✅ Sentiment model found in cache!", severity="success")
                await self.load_model_from_cache()
                return True
                
        except (LocalEntryNotFoundError, HFValidationError):
            self.app.notify("⚠️ Sentiment model not found in cache", severity="warning")
            self.show_download_button()
            return False
        except Exception as e:
            self.app.notify(f"⚠️ Error checking model availability: {str(e)}", severity="error")
            self.show_download_button()
            return False
    
    async def load_model_from_cache(self):
        """Load the model from the local cache asynchronously."""
        try:
            # Load tokenizer and model in separate threads to avoid blocking
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, 
                MODEL_ID, 
                local_files_only=True
            )
            
            self.model = await asyncio.to_thread(
                AutoModelForSequenceClassification.from_pretrained, 
                MODEL_ID, 
                local_files_only=True
            )
            
            self.model_loaded = True
            self.hide_download_button()
            self.app.notify("✅ Sentiment model loaded successfully!", severity="success")
            return True
        except Exception as e:
            self.app.notify(f"❌ Failed to load cached model: {str(e)}", severity="error")
            self.model_loaded = False
            self.show_download_button()
            return False
    
    def show_download_button(self):
        """Show the download button."""
        download_button = self.query_one("#download_model_button", Button)
        download_button.display = True
    
    def hide_download_button(self):
        """Hide the download button."""
        download_button = self.query_one("#download_model_button", Button)
        download_button.display = False
    
    # def show_download_progress(self):
    #     """Show the download progress indicator."""
    #     progress = self.query_one("#download_progress", LoadingIndicator)
    #     progress.display = True
    
    # def hide_download_progress(self):
    #     """Hide the download progress indicator."""
    #     progress = self.query_one("#download_progress", LoadingIndicator)
    #     progress.display = False
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button press events."""
        if event.button.id == "search_button":
            search_input = self.query_one("#search_input", Input)
            search_query = search_input.value.strip()
            if search_query:
                asyncio.create_task(self.fetch_videos(search_query))
            else:
                self.app.notify("⚠ Please enter a search keyword!", severity="warning")
        elif event.button.id == "download_model_button":
            # if not self.download_in_progress:
            asyncio.create_task(self.download_sentiment_model())
    
    async def download_sentiment_model(self):
        """Download the sentiment analysis model."""
        if self.download_in_progress:
            self.app.notify("⚠️ Download already in progress", severity="warning")
            return
            
        self.download_in_progress = True
        download_button = self.query_one("#download_model_button", Button)
        # download_button.disabled = True
        
        self.app.notify("🔄 Downloading sentiment model... This may take a few minutes.", severity="information")
        # self.show_download_progress()
        
        try:
            def build_sentiment_pipeline():
                return pipeline(
                    "text-classification",
                    model=MODEL_ID,
                    tokenizer=MODEL_ID,
                )
            # Download the model (this will automatically cache it)
            sentiment_analyzer = build_sentiment_pipeline() # pipeline creates a subprocess, so no need to create a new thread for it.
            self.tokenizer = sentiment_analyzer.tokenizer
            self.model = sentiment_analyzer.model
            
            self.model_loaded = True
            self.app.notify("✅ Model downloaded and loaded successfully!", severity="success")
            self.hide_download_button()
            
        except Exception as e:
            self.app.notify(f"❌ Failed to download sentiment model: {str(e)}", severity="error")
            download_button.disabled = False
            self.model_loaded = False
        finally:
            self.download_in_progress = False
            # self.hide_download_progress()

    async def fetch_videos(self, search_query):
        """Fetch YouTube videos based on search query and update the option list."""
        self.app.notify("🔍 Searching YouTube... Please wait.", severity="information")
        video_list_widget = self.query_one("#video_list", OptionList)
        video_list_widget.clear_options()

        try:
            # ✅ Fetch YouTube results asynchronously
            results = await asyncio.to_thread(YoutubeSearch, search_query, max_results=15)
            results_json = await asyncio.to_thread(results.to_json)
            video_data = json.loads(results_json)["videos"]

            # ✅ Filter videos less than 3 minutes
            filtered_videos = [
                {"id": video["id"], "title": video["title"]}
                for video in video_data if await asyncio.to_thread(
                    self.duration_to_seconds, video.get("duration", "0")
                ) < 180
            ]

            if not filtered_videos:
                self.app.notify("⚠ No short videos found under 3 minutes!", severity="warning")
                return

            # ✅ Show the video list UI component
            self.query_one("#video_label", Static).remove_class("hidden")

            # ✅ Populate the video list
            for video in filtered_videos:
                video_list_widget.add_option(f"{video['title']} ({video['id']})")

            self.app.notify(f"✅ Found {len(filtered_videos)} short videos!", severity="success")

        except Exception as e:
            self.app.notify(f"❌ Error fetching videos: {e}", severity="error")

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        """Fetch transcript when a video is selected."""
        selected_option = event.option.prompt  # ✅ Get video ID directly from OptionList
        video_id = selected_option.split("(")[-1].strip(")")

        self.app.notify(f"📜 Fetching transcript for Video ID: {video_id}...", severity="information")
        # ✅ Fetch transcript asynchronously
        transcript_text = await self.get_transcript(video_id)

        # ✅ Display transcript using the new method
        self.display_transcript(transcript_text)

        # ✅ Call Sentiment Analysis on the transcript
        if self.model_loaded:
            await self.display_sentiment_report(transcript_text)
        else:
            self.app.notify("⚠️ Sentiment analysis unavailable - model not loaded", severity="warning")
            self.show_download_button()

    def display_transcript(self, transcript_text):
        """Handles displaying the transcript UI."""
        self.query_one("#transcript_label", Static).remove_class("hidden")

        # ✅ Update transcript display
        transcript_display = self.query_one("#transcript_display", Static)
        transcript_display.update(transcript_text)

    async def get_transcript(self, video_id):
        """Fetch transcript for a given video ID with enhanced error handling."""
        try:
            # ✅ Fetch transcript asynchronously using a thread
            transcript_data = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id)

            if not transcript_data:
                self.app.notify("⚠ No transcript data found.", severity="warning")
                return "⚠ Transcript not available."

            # ✅ Convert transcript list to readable text
            transcript_text = " ".join(entry["text"] for entry in transcript_data)
            self.app.notify("✅ Transcript fetched successfully!", severity="success")
            return transcript_text

        except Exception as e:
            # ✅ Catch unexpected errors and notify user
            self.app.notify(f"⚠ Error fetching transcript: {e}", severity="error")
            return "⚠ Transcript not available."

    async def display_sentiment_report(self, text):
        """Perform sentiment analysis and display the results."""
        if not self.model_loaded:
            self.app.notify("⚠ Model not loaded. Please download the model first.", severity="warning")
            self.show_download_button()
            return
        
        if not text or text == "⚠ Transcript not available.":
            self.app.notify("⚠ No transcript available for sentiment analysis.", severity="warning")
            return

        # ✅ Run Sentiment Analysis
        self.app.notify(f"Starting Sentiment Analysis", severity="success")

        sentiment_report = await self.analyze_sentiment(text)

        # ✅ Show sentiment analysis UI component
        self.query_one("#sentiment_label", Static).remove_class("hidden")

        # ✅ Update sentiment display
        sentiment_display = self.query_one("#sentiment_display", Static)
        sentiment_display.update(
            f"🔹 **Sentiment:** {sentiment_report['sentiment']} \n"
            f"🔹 **Confidence:** {sentiment_report['confidence']:.2f} \n\n"
            f"🔹 **Detailed Analysis:** \n{sentiment_report['gemini_report']}"
        )

    async def analyze_sentiment(self, text):
        """Perform sentiment analysis using RoBERTa and Gemini AI."""
        if not text:
            return {"error": "No text provided for sentiment analysis"}

        # ✅ Step 1: Run RoBERTa for Sentiment Prediction asynchronously
        try:
            # Tokenize the text
            inputs = await asyncio.to_thread(
                self.tokenizer, 
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Get model outputs asynchronously
            def run_model():
                with torch.no_grad():
                    return self.model(**inputs)
                
            outputs = await asyncio.to_thread(run_model)
            
            # Process results
            scores = outputs.logits.softmax(dim=1).tolist()[0]
            labels = ["Negative", "Neutral", "Positive"]
            sentiment = labels[scores.index(max(scores))]
            confidence = max(scores)
            
        except Exception as e:
            self.app.notify(f"❌ RoBERTa sentiment analysis failed: {str(e)}", severity="error")
            return {
                "sentiment": "Error", 
                "confidence": 0.0, 
                "gemini_report": f"⚠ RoBERTa sentiment analysis failed: {str(e)}"
            }

        # ✅ Step 2: Generate Detailed Report using Gemini AI asynchronously
        api_key = await asyncio.to_thread(self.fetch_gemini_api_key)
        if not api_key:
            return {"sentiment": sentiment, "confidence": confidence, "gemini_report": "⚠ Gemini AI Key missing."}

        gemai.configure(api_key=api_key)
        try:
            model = gemai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Perform a detailed sentiment analysis of the following text: 
            "{text}"
            Include sentiment classification, reasoning, tone, and emotional indicators.
            """
            response = await asyncio.to_thread(model.generate_content, prompt)
            gemini_report = response.text
        except Exception as e:
            gemini_report = f"⚠ Gemini AI sentiment analysis failed: {str(e)}"

        # ✅ Return Combined Sentiment Results
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "gemini_report": gemini_report
        }

    async def fetch_gemini_api_key(self):
        """Fetch Gemini API Key from settings.json file asynchronously."""
        try:
            # Open and read settings file in a separate thread
            file_content = await asyncio.to_thread(open, SETTINGS_FILE, "r")
            content = await asyncio.to_thread(file_content.read)
            await asyncio.to_thread(file_content.close)
            
            # Parse JSON in a separate thread
            settings = await asyncio.to_thread(json.loads, content)

            # ✅ Navigate to the correct API key location
            api_key = settings.get("data_sources", {}).get("genai_model", {}).get("apikey", None)

            if not api_key:
                self.app.notify("⚠ Gemini API Key not found in settings.", severity="warning")
                return None

            return api_key

        except Exception as e:
            self.app.notify(f"❌ Unexpected error loading API key: {e}", severity="error")
            return None

    def duration_to_seconds(self, duration):
        """Convert duration string (MM:SS) to seconds, handling possible integer values."""
        if isinstance(duration, int):
            return duration  # Already in seconds

        if not isinstance(duration, str):
            return 0  # Handle unexpected format

        parts = duration.split(":")
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        return int(parts[0]) if parts[0].isdigit() else 0