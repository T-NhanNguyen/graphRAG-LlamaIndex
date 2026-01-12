import math
import json
import os
import logging
import re
from collections import Counter
from typing import List, Dict, Any, Optional
from graphrag_config import settings

logger = logging.getLogger(__name__)

class GarbageFilter:
    """
    Multi-stage filtering for knowledge graph chunks.
    Implements preprocessing (stateless) and advanced (stateful/LLM) filters.
    """

    @staticmethod
    def calculateRepetitionRatio(text: str) -> float:
        """
        Calculate the ratio of character repetition in text.
        Ignores whitespace to avoid false positives on highly indented/formatted text.
        """
        if not text:
            return 0.0
        # Filter out whitespace before counting
        contentOnly = "".join(char for char in text if not char.isspace())
        if not contentOnly:
            return 0.0
            
        charCounts = Counter(contentOnly)
        maxRepetitionCount = max(charCounts.values()) if charCounts else 0
        repetitionRatio = maxRepetitionCount / len(contentOnly)
        return repetitionRatio

    @staticmethod
    def calculateEntropy(text: str) -> float:
        """Calculate Shannon entropy for a string to measure information density."""
        if not text:
            return 0.0
        charCounts = Counter(text)
        totalChars = len(text)
        shannonEntropy = -sum((count / totalChars) * math.log2(count / totalChars) for count in charCounts.values())
        return shannonEntropy

    @staticmethod
    def calculateMalformedRatio(text: str) -> float:
        """Detect broken ligatures (e.g., 'fi ', 'fl ') common in poor PDF extraction."""
        if not text:
            return 0.0
        # Common broken ligatures in English (f+i, f+l, f+f, etc.)
        brokenPatterns = [r'f\s+i', r'f\s+l', r'f\s+f']
        brokenLigatureCount = 0
        for pattern in brokenPatterns:
            brokenLigatureCount += len(re.findall(pattern, text))
        
        wordsInText = text.split()
        if not wordsInText:
            return 0.0
        brokenLigatureRatio = brokenLigatureCount / len(wordsInText)
        return brokenLigatureRatio

    @staticmethod
    def calculateWhitespaceDensity(text: str) -> float:
        """
        Calculate the ratio of whitespace characters to total text length.
        Useful for identifying 'diluted' chunks where formatting noise outweighs content.
        """
        if not text:
            return 0.0
        totalChars = len(text)
        whitespaceCount = sum(1 for char in text if char.isspace())
        densityRatio = whitespaceCount / totalChars
        return densityRatio

    def isGarbagePre(self, text: str) -> Optional[str]:
        """
        Run fast, deterministic preprocessing filters on a text chunk.
        
        Args:
            text: The text chunk to evaluate.
            
        Returns:
            A string describing the failure reason if it's garbage, else None.
        """
        # 1. Repetition Filter: Catches character 'stuttering' or line noise
        repetitionRatio = self.calculateRepetitionRatio(text)
        if repetitionRatio > settings.FILTER_REPETITION_THRESHOLD:
            return f"Repetition too high: {repetitionRatio:.2f} > {settings.FILTER_REPETITION_THRESHOLD}"

        # 2. Entropy Filter: Catches low-information repetitive text or high-entropy random noise
        shannonEntropy = self.calculateEntropy(text)
        if shannonEntropy < settings.FILTER_MIN_ENTROPY:
            return f"Entropy too low: {shannonEntropy:.2f} < {settings.FILTER_MIN_ENTROPY}"
        if shannonEntropy > settings.FILTER_MAX_ENTROPY:
            return f"Entropy too high: {shannonEntropy:.2f} > {settings.FILTER_MAX_ENTROPY}"

        # 3. Malformed Text Filter: Detects common OCR/PDF artifacts like broken ligatures
        brokenLigatureRatio = self.calculateMalformedRatio(text)
        if brokenLigatureRatio > settings.FILTER_MALFORMED_THRESHOLD:
            return f"Malformed formatting: {brokenLigatureRatio:.2f} > {settings.FILTER_MALFORMED_THRESHOLD}"

        # 4. Whitespace Density Filter: Catches 'diluted' chunks drowned in whitespace
        whitespaceDensity = self.calculateWhitespaceDensity(text)
        if whitespaceDensity > settings.FILTER_MAX_WHITESPACE_DENSITY:
            return f"Whitespace too dense: {whitespaceDensity:.2f} > {settings.FILTER_MAX_WHITESPACE_DENSITY}"

        return None

class GarbageLogger:
    """Utility to log skipped garbage chunks for technical tracking and manual audit."""
    
    def __init__(self, logPath: str = None):
        self.logPath = logPath or os.path.join(settings.OUTPUT_DIR, "pruning_log.json")
        self.evidenceDir = os.path.join(settings.OUTPUT_DIR, "pruning_evidence")
        os.makedirs(os.path.dirname(self.logPath), exist_ok=True)
        os.makedirs(self.evidenceDir, exist_ok=True)
        self.prunedLogs = []

    def log(self, chunkId: str, text: str, reason: str, metadata: Dict[str, Any] = None):
        """Append a garbage chunk to the log and save an individual file for inspection."""
        logEntry = {
            "chunkId": chunkId,
            "reason": reason,
            "text": text,
            "metadata": metadata or {}
        }
        self.prunedLogs.append(logEntry)
        self._persistToDisk()
        self._saveIndividualEvidence(chunkId, text, reason)

    def _saveIndividualEvidence(self, chunkId: str, text: str, reason: str):
        """Save full pruned text to an individual Markdown file for easy auditing."""
        try:
            # Create a clean filename from the reason
            safeReason = "".join(char for char in reason[:20] if char.isalnum() or char in " -_").strip()
            evidenceFilename = f"{chunkId}_{safeReason}.md"
            evidenceFilePath = os.path.join(self.evidenceDir, evidenceFilename)
            
            with open(evidenceFilePath, 'w', encoding='utf-8') as evidenceFile:
                evidenceFile.write(f"REASON: {reason}\n")
                evidenceFile.write("-" * 40 + "\n")
                evidenceFile.write(text)
        except Exception as exc:
            logger.error(f"Failed to save individual pruned file: {exc}")

    def _persistToDisk(self):
        """Save logs to disk in both JSON (data) and JS (viewer) formats."""
        try:
            # 1. Standard JSON log for processing
            with open(self.logPath, 'w', encoding='utf-8') as jsonFile:
                json.dump(self.prunedLogs, jsonFile, indent=2)
            
            # 2. JS Wrapper for the interactive viewer (bypasses CORS)
            jsLogPath = self.logPath.replace(".json", ".js")
            with open(jsLogPath, 'w', encoding='utf-8') as jsFile:
                jsFile.write(f"window.PRUNING_DATA = {json.dumps(self.prunedLogs, indent=2)};")
            
            logger.debug(f"Saved pruning logs to {self.logPath}")
        except Exception as exc:
            logger.error(f"Failed to save garbage logs: {exc}")

# Singleton instances for global access
garbageFilter = GarbageFilter()
garbageLogger = GarbageLogger()
