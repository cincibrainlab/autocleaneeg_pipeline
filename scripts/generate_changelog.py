#!/usr/bin/env python3
"""
Automated changelog generation script for AutoClean EEG Pipeline.

This script generates a changelog based on git commit history, categorizing
commits by type and providing a standardized format for release notes.
"""

import argparse
import subprocess
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class ChangelogGenerator:
    """Generate changelog from git commit history."""
    
    def __init__(self, repo_path: Path = None):
        """Initialize changelog generator."""
        self.repo_path = repo_path or Path.cwd()
        self.commit_patterns = {
            'features': [
                r'^feat(\([^)]*\))?!?:\s*(.+)',
                r'^add(\([^)]*\))?!?:\s*(.+)',
                r'^feature(\([^)]*\))?!?:\s*(.+)',
            ],
            'fixes': [
                r'^fix(\([^)]*\))?!?:\s*(.+)',
                r'^bug(\([^)]*\))?!?:\s*(.+)',
                r'^patch(\([^)]*\))?!?:\s*(.+)',
            ],
            'improvements': [
                r'^improve(\([^)]*\))?!?:\s*(.+)',
                r'^enhance(\([^)]*\))?!?:\s*(.+)',
                r'^update(\([^)]*\))?!?:\s*(.+)',
                r'^refactor(\([^)]*\))?!?:\s*(.+)',
                r'^perf(\([^)]*\))?!?:\s*(.+)',
            ],
            'docs': [
                r'^docs(\([^)]*\))?!?:\s*(.+)',
                r'^doc(\([^)]*\))?!?:\s*(.+)',
            ],
            'tests': [
                r'^test(\([^)]*\))?!?:\s*(.+)',
                r'^tests(\([^)]*\))?!?:\s*(.+)',
            ],
            'ci': [
                r'^ci(\([^)]*\))?!?:\s*(.+)',
                r'^build(\([^)]*\))?!?:\s*(.+)',
            ],
            'breaking': [
                r'^.*!:\s*(.+)',  # Any commit with ! indicates breaking change
            ]
        }
    
    def get_git_tags(self) -> List[str]:
        """Get all git tags sorted by version."""
        try:
            result = subprocess.run(
                ['git', 'tag', '--sort=-version:refname'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return [tag.strip() for tag in result.stdout.split('\n') if tag.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def get_commits_between(self, from_ref: Optional[str], to_ref: str = 'HEAD') -> List[str]:
        """Get commit messages between two git references."""
        if from_ref:
            commit_range = f"{from_ref}..{to_ref}"
        else:
            commit_range = to_ref
        
        try:
            result = subprocess.run(
                ['git', 'log', '--pretty=format:%s', commit_range],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return [commit.strip() for commit in result.stdout.split('\n') if commit.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def categorize_commit(self, commit_message: str) -> Tuple[str, str]:
        """Categorize a commit message and extract the description."""
        for category, patterns in self.commit_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, commit_message, re.IGNORECASE)
                if match:
                    # Extract the description (last capturing group)
                    description = match.groups()[-1] if match.groups() else commit_message
                    return category, description
        
        return 'other', commit_message
    
    def generate_section(self, category: str, commits: List[str]) -> str:
        """Generate a changelog section for a category."""
        if not commits:
            return ""
        
        section_titles = {
            'breaking': '💥 Breaking Changes',
            'features': '✨ New Features', 
            'improvements': '🚀 Improvements',
            'fixes': '🐛 Bug Fixes',
            'docs': '📚 Documentation',
            'tests': '🧪 Tests',
            'ci': '🔧 CI/CD',
            'other': '📝 Other Changes'
        }
        
        title = section_titles.get(category, f"📝 {category.title()}")
        section = f"\n### {title}\n\n"
        
        for commit in commits:
            # Clean up commit message
            commit = commit.strip()
            if not commit.endswith('.'):
                commit += '.'
            section += f"- {commit}\n"
        
        return section
    
    def generate_changelog_entry(self, version: str, from_tag: Optional[str] = None, 
                                to_ref: str = 'HEAD') -> str:
        """Generate a changelog entry for a version."""
        commits = self.get_commits_between(from_tag, to_ref)
        
        if not commits:
            return f"\n## {version} ({datetime.now().strftime('%Y-%m-%d')})\n\nNo changes in this release.\n"
        
        # Categorize commits
        categorized = {category: [] for category in self.commit_patterns.keys()}
        categorized['other'] = []
        
        for commit in commits:
            category, description = self.categorize_commit(commit)
            categorized[category].append(description)
        
        # Generate changelog entry
        changelog = f"\n## {version} ({datetime.now().strftime('%Y-%m-%d')})\n"
        
        # Order sections by importance
        section_order = ['breaking', 'features', 'improvements', 'fixes', 'docs', 'tests', 'ci', 'other']
        
        for category in section_order:
            if categorized[category]:
                changelog += self.generate_section(category, categorized[category])
        
        return changelog
    
    def update_changelog_file(self, changelog_entry: str, changelog_file: Path = None):
        """Update the CHANGELOG.md file with a new entry."""
        if changelog_file is None:
            changelog_file = self.repo_path / "CHANGELOG.md"
        
        # Read existing changelog
        existing_content = ""
        if changelog_file.exists():
            with open(changelog_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Create new content
        if existing_content:
            # Insert new entry after the header
            lines = existing_content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('## ') and i > 0:
                    header_end = i
                    break
            
            if header_end > 0:
                new_content = '\n'.join(lines[:header_end]) + '\n' + changelog_entry + '\n' + '\n'.join(lines[header_end:])
            else:
                new_content = changelog_entry + '\n' + existing_content
        else:
            # Create new changelog
            header = "# Changelog\n\nAll notable changes to AutoClean EEG Pipeline will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n"
            new_content = header + changelog_entry
        
        # Write updated content
        with open(changelog_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Changelog updated: {changelog_file}")
    
    def generate_release_notes(self, version: str, from_tag: Optional[str] = None) -> str:
        """Generate release notes for a specific version."""
        commits = self.get_commits_between(from_tag, 'HEAD')
        
        if not commits:
            return f"## Release {version}\n\nNo changes in this release."
        
        # Categorize commits
        categorized = {category: [] for category in self.commit_patterns.keys()}
        categorized['other'] = []
        
        for commit in commits:
            category, description = self.categorize_commit(commit)
            categorized[category].append(description)
        
        # Generate release notes
        notes = f"## Release {version}\n"
        
        # Summary
        total_commits = len(commits)
        feature_count = len(categorized['features'])
        fix_count = len(categorized['fixes'])
        
        notes += f"\n**{total_commits} commits** with {feature_count} new features and {fix_count} bug fixes.\n"
        
        # Highlight breaking changes
        if categorized['breaking']:
            notes += "\n⚠️ **This release contains breaking changes!**\n"
        
        # Add sections
        section_order = ['breaking', 'features', 'improvements', 'fixes', 'docs', 'tests', 'ci', 'other']
        
        for category in section_order:
            if categorized[category]:
                notes += self.generate_section(category, categorized[category])
        
        return notes


def main():
    """Main entry point for changelog generation."""
    parser = argparse.ArgumentParser(description="Generate changelog for AutoClean EEG Pipeline")
    parser.add_argument('version', help='Version number (e.g., 1.4.2)')
    parser.add_argument('--from-tag', help='Starting tag for changelog generation')
    parser.add_argument('--to-ref', default='HEAD', help='Ending reference (default: HEAD)')
    parser.add_argument('--output', help='Output file path (default: CHANGELOG.md)')
    parser.add_argument('--release-notes', action='store_true', 
                       help='Generate release notes instead of changelog entry')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Print changelog without updating file')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ChangelogGenerator()
    
    # Auto-detect from tag if not provided
    if not args.from_tag:
        tags = generator.get_git_tags()
        if tags:
            args.from_tag = tags[0]
            print(f"🔍 Auto-detected previous tag: {args.from_tag}")
        else:
            print("⚠️ No previous tags found, generating changelog from all commits")
    
    # Generate changelog or release notes
    if args.release_notes:
        content = generator.generate_release_notes(args.version, args.from_tag)
        output_file = Path(args.output) if args.output else Path('RELEASE_NOTES.md')
    else:
        content = generator.generate_changelog_entry(args.version, args.from_tag, args.to_ref)
        output_file = Path(args.output) if args.output else Path('CHANGELOG.md')
    
    # Output
    if args.dry_run:
        print("📋 Generated content:")
        print("=" * 50)
        print(content)
        print("=" * 50)
    else:
        if args.release_notes:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Release notes written to: {output_file}")
        else:
            generator.update_changelog_file(content, output_file)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())