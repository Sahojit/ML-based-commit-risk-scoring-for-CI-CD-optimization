"""
Git Data Extractor
Extracts commit metadata from GitHub repositories using PyGithub
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from github import Github, Auth, GithubException
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitExtractor:
    """
    Extracts commit data from GitHub repositories
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitExtractor
        
        Args:
            github_token: GitHub personal access token (optional)
        """
        # Get token from parameter or environment
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        
        if not self.token or self.token == "ghp_PLACEHOLDER":
            logger.warning("No valid GitHub token provided. Rate limits will be restricted.")
            self.github = Github()  # Anonymous access (60 requests/hour)
        else:
            # Use new authentication method (fixes deprecation warning)
            auth = Auth.Token(self.token)
            self.github = Github(auth=auth)  # Authenticated (5000 requests/hour)
            logger.info("GitHub API authenticated successfully")
        
        # Check rate limit (fixed attribute access)
        try:
            rate_limit = self.github.get_rate_limit()
            core_limit = rate_limit.core
            logger.info(f"GitHub API rate limit: {core_limit.remaining}/{core_limit.limit}")
        except Exception as e:
            logger.warning(f"Could not check rate limit: {e}")
    
    def extract_commits(
        self,
        repo_owner: str,
        repo_name: str,
        max_commits: int = 1000,
        branch: str = "main"
    ) -> pd.DataFrame:
        """
        Extract commit data from a GitHub repository
        
        Args:
            repo_owner: Repository owner (user or organization)
            repo_name: Repository name
            max_commits: Maximum number of commits to fetch
            branch: Branch name to analyze
        
        Returns:
            DataFrame with commit data
        """
        logger.info(f"Starting data extraction from {repo_owner}/{repo_name}")
        logger.info(f"Branch: {branch}, Max commits: {max_commits}")
        
        try:
            # Get repository
            repo = self.github.get_repo(f"{repo_owner}/{repo_name}")
            logger.info(f"Repository found: {repo.full_name}")
            
            # Get commits
            commits_data = []
            commits = repo.get_commits(sha=branch)
            
            logger.info("Fetching commits...")
            
            for idx, commit in enumerate(commits[:max_commits]):
                if idx % 100 == 0:
                    logger.info(f"Processed {idx} commits...")
                
                try:
                    # Extract commit metadata
                    commit_dict = self._extract_commit_data(commit)
                    commits_data.append(commit_dict)
                    
                except Exception as e:
                    logger.error(f"Error processing commit {commit.sha}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(commits_data)} commits")
            
            # Convert to DataFrame
            df = pd.DataFrame(commits_data)
            
            # Add extraction timestamp
            df['extracted_at'] = datetime.now()
            
            return df
        
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def _extract_commit_data(self, commit) -> Dict:
        """
        Extract data from a single commit
        
        Args:
            commit: PyGithub commit object
        
        Returns:
            Dictionary with commit data
        """
        # Get commit statistics
        stats = commit.stats
        
        # Count files changed (convert PaginatedList to list first)
        try:
            files_changed = commit.files.totalCount
        except:
            # Fallback: convert to list and count
            files_changed = len(list(commit.files))
        
        # Extract data
        commit_data = {
            'commit_hash': commit.sha,
            'author': commit.commit.author.email if commit.commit.author else 'unknown',
            'author_name': commit.commit.author.name if commit.commit.author else 'unknown',
            'timestamp': commit.commit.author.date,
            'message': commit.commit.message,
            'files_changed': files_changed,  # ← FIXED!
            'lines_added': stats.additions,
            'lines_deleted': stats.deletions,
            'total_changes': stats.total
        }
        
        return commit_data
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            output_path: Path to output CSV file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        logger.info(f"Total records: {len(df)}")
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get basic statistics about extracted data
        
        Args:
            df: DataFrame with commit data
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_commits': len(df),
            'unique_authors': df['author'].nunique(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'avg_files_changed': df['files_changed'].mean(),
            'avg_lines_added': df['lines_added'].mean(),
            'avg_lines_deleted': df['lines_deleted'].mean(),
            'total_lines_added': df['lines_added'].sum(),
            'total_lines_deleted': df['lines_deleted'].sum()
        }
        
        return stats


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize extractor
    extractor = GitExtractor()
    
    # Example: Extract from a public repository
    repo_owner = "pandas-dev"
    repo_name = "pandas"
    
    logger.info("=" * 70)
    logger.info("TESTING GIT EXTRACTOR")
    logger.info("=" * 70)
    
    try:
        # Extract commits (limit to 10 for testing)
        df = extractor.extract_commits(
            repo_owner=repo_owner,
            repo_name=repo_name,
            max_commits=10,
            branch="main"
        )
        
        # Display results
        print("\n✅ Extraction successful!")
        print(f"\nFirst 5 commits:")
        print(df[['commit_hash', 'author_name', 'timestamp', 'files_changed']].head())
        
        # Get statistics
        stats = extractor.get_statistics(df)
        print(f"\n📊 Statistics:")
        print(f"   Total commits: {stats['total_commits']}")
        print(f"   Unique authors: {stats['unique_authors']}")
        print(f"   Avg files changed: {stats['avg_files_changed']:.2f}")
        print(f"   Avg lines added: {stats['avg_lines_added']:.2f}")
        print(f"   Avg lines deleted: {stats['avg_lines_deleted']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")