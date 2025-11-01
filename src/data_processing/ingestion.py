from typing import AsyncIterator, Dict, List, Tuple
import asyncpraw
import asyncprawcore
from src.utils.config import load_credentials
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def fetch_reddit_data(subreddit_names: List[str], limit: int = 100) -> AsyncIterator[Tuple[Dict, List[Dict]]]:
    """
    Connects to Reddit asynchronously and fetches top posts AND their top-level comments efficiently.
    """
    credentials = load_credentials()
    async with asyncpraw.Reddit(
        client_id=credentials["reddit_client_id"],
        client_secret=credentials["reddit_client_secret"],
        user_agent=credentials["reddit_user_agent"],
    ) as reddit:
        for sub_name in subreddit_names:
            logging.info(f"Inizio fetch per subreddit: r/{sub_name}")
            post_counter = 0
            try:
                subreddit = await reddit.subreddit(sub_name)
                async for post in subreddit.top(time_filter="year", limit=limit):
                    if post_counter >= limit:
                        break
                    post_counter += 1

                    author_name = 'deleted'
                    if post.author:
                        author_name = post.author.name
                    
                    post_data = {
                        "id": post.id,
                        "author": author_name,
                        "content": f"{post.title} {post.selftext}",
                        "timestamp": int(post.created_utc),
                        "score": post.score,
                        "subreddit": post.subreddit.display_name,
                    }

                    comments_data = []
                    
                    # Salta post senza commenti
                    if post.num_comments == 0:
                        yield post_data, comments_data
                        continue
                    
                    try:
                        post.comment_limit = 20  # Carica solo i primi 20 commenti top
                        post.comment_sort = "top"  # Ordina per score
                        
                        # Forza il caricamento accedendo ai commenti
                        _ = await post.comments()
                        
                        top_comments = []
                        for comment in post.comments:
                            if isinstance(comment, asyncpraw.models.Comment):
                                top_comments.append(comment)
                        
                        num_comments = len(top_comments)
                        k = 10 if num_comments > 50 else 5
                        
                        for comment in top_comments[:k]:
                            comment_author = 'deleted'
                            if comment.author:
                                comment_author = comment.author.name
                            
                            if not hasattr(comment, 'body') or not comment.body or comment.body in ['[deleted]', '[removed]']:
                                continue

                            comments_data.append({
                                "id": comment.id,
                                "author": comment_author,
                                "content": comment.body,
                                "timestamp": int(comment.created_utc),
                                "score": comment.score,
                                "post_id": post.id
                            })
                            
                    except Exception as e:
                        logging.warning(f"Errore caricamento commenti per post {post.id}: {e}")
                        pass

                    yield post_data, comments_data
                    
                    if post_counter % 20 == 0:
                        logging.info(f"r/{sub_name}: {post_counter}/{limit} post processati - {len(comments_data)} commenti estratti")
                        
            except asyncprawcore.exceptions.AsyncPrawcoreException as e:
                logging.error(f"Errore durante il fetch da r/{sub_name}: {e}")
                continue

            logging.info(f"Fetch completato per r/{sub_name}. Post processati: {post_counter}")