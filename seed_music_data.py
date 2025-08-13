"""
Seed music catalog data into the vector database.
"""
import os
from dotenv import load_dotenv
from config import db, vector_store

load_dotenv()

def seed_music_catalog():
    """Extract music data from SQL database and populate vector store."""
    print("Seeding music catalog into vector database...")
    
    # Query to get comprehensive music data
    music_query = """
    SELECT 
        t."Name" as track_name,
        a."Name" as artist_name,
        al."Title" as album_title,
        g."Name" as genre,
        t."Composer",
        t."Milliseconds",
        t."UnitPrice"
    FROM "Track" t
    JOIN "Album" al ON t."AlbumId" = al."AlbumId"
    JOIN "Artist" a ON al."ArtistId" = a."ArtistId"
    LEFT JOIN "Genre" g ON t."GenreId" = g."GenreId"
    ORDER BY a."Name", al."Title", t."TrackId"
    """
    
    try:
        # Get music data from database
        result = db.run(music_query)
        
        if not result or result == "[]":
            print("No music data found in database")
            return
            
        import json
        music_data = json.loads(result)
        print(f"Found {len(music_data)} tracks to process")
        
        # Prepare documents for vector store
        documents = []
        metadatas = []
        
        for track in music_data:
            # Create searchable text content
            content = f"""
Track: {track['track_name']}
Artist: {track['artist_name']}
Album: {track['album_title']}
Genre: {track.get('genre', 'Unknown')}
Composer: {track.get('composer', 'Unknown')}
Duration: {track.get('milliseconds', 0)} ms
Price: ${track.get('unitprice', 0)}
            """.strip()
            
            documents.append(content)
            metadatas.append({
                'track_name': track['track_name'],
                'artist_name': track['artist_name'],
                'album_title': track['album_title'],
                'genre': track.get('genre', 'Unknown'),
                'price': track.get('unitprice', 0)
            })
        
        # Add to vector store in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            
            vector_store.add_texts(
                texts=batch_docs,
                metadatas=batch_metas
            )
            
            print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # Persist the changes
        vector_store.persist()
        print(f"Successfully seeded {len(documents)} tracks into vector database")
        
        # Test the vector store
        test_results = vector_store.similarity_search("nirvana", k=3)
        print(f"\nTest search for 'nirvana' returned {len(test_results)} results")
        for result in test_results[:2]:
            print(f"- {result.page_content.split('Track: ')[1].split('Artist: ')[0].strip()}")
            
    except Exception as e:
        print(f"Error seeding music catalog: {e}")

if __name__ == "__main__":
    seed_music_catalog()