from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
UNITY_ASSETS = ROOT / "tasks" / "task_template_designer" / "Assets"
BUILD_SOURCE = UNITY_ASSETS / "Scripts" / "Editor" / "NovPhyBuild.cs"
COMPILER_RESPONSE = UNITY_ASSETS / "csc.rsp"


class DeterministicUnityBuildContractTests(unittest.TestCase):
    def test_roslyn_compilation_is_deterministic(self) -> None:
        # Given: the Unity project-level Roslyn response file.
        response_tokens = COMPILER_RESPONSE.read_text(encoding="ascii").split()

        # When/Then: Unity receives exactly its supported deterministic compiler option.
        self.assertEqual(response_tokens, ["-deterministic"])

    def test_player_build_omits_unique_identity_and_keeps_strict_failures(self) -> None:
        # Given: the build-player options initializer consumed by Unity.
        source = BUILD_SOURCE.read_text(encoding="utf-8")
        initializer = re.search(
            r"new\s+BuildPlayerOptions\s*\{(?P<body>.*?)\}\s*;",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(initializer, "BuildPlayerOptions initializer is missing")
        expression = re.search(r"\boptions\s*=\s*([^,}\n]+)", initializer.group("body"))
        self.assertIsNotNone(expression, "BuildPlayerOptions.options assignment is missing")

        # When: the combinable BuildOptions members are parsed structurally.
        option_names = {
            token.strip()
            for token in expression.group(1).split("|")
            if token.strip()
        }

        # Then: generated player identity is suppressed without weakening strict mode.
        self.assertEqual(
            option_names,
            {"BuildOptions.NoUniqueIdentifier", "BuildOptions.StrictMode"},
        )


if __name__ == "__main__":
    unittest.main()
