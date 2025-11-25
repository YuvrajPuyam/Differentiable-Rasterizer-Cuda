import torch

def generate_icosphere(level=2, device='cpu'):
    """
    Fully vectorized Icosphere Generator.
    0 loops over faces. Runs instantly on GPU/CPU.
    """
    # 1. Base Icosahedron
    t = (1.0 + 5.0**0.5) / 2.0
    verts = torch.tensor([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=torch.float32, device=device)
    
    faces = torch.tensor([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=torch.long, device=device)

    # 2. Vectorized Subdivision
    for _ in range(level):
        # Extract the three vertices for all faces at once: (F, 3) -> (F, 3, 3)
        tri_verts = verts[faces] 
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]

        # Calculate midpoints for all edges: (F, 3)
        mid_01 = (v0 + v1) / 2.0
        mid_12 = (v1 + v2) / 2.0
        mid_20 = (v2 + v0) / 2.0

        # Stack all vertices (original + new midpoints) into a huge list
        # We need to reconstruct faces:
        # T0: v0, m01, m20
        # T1: v1, m12, m01
        # T2: v2, m20, m12
        # T3: m01, m12, m20
        
        # We'll create a new vertex list that is simply ALL vertices used by ALL new faces
        # This creates duplicates, but we will merge them later.
        # Shape: (4 * F, 3, 3) -> flattened to (12 * F, 3)
        
        # Construct the 4 new triangles per face
        # (F, 3, 3) stack
        new_faces_verts = torch.cat([
            torch.stack([v0, mid_01, mid_20], dim=1),      # Top
            torch.stack([v1, mid_12, mid_01], dim=1),      # Right
            torch.stack([v2, mid_20, mid_12], dim=1),      # Left
            torch.stack([mid_01, mid_12, mid_20], dim=1)   # Center
        ], dim=0) # (4F, 3, 3)

        # Flatten to (Total_Vertices, 3)
        flat_verts = new_faces_verts.view(-1, 3)

        # 3. Merge Duplicates (The Magic Step)
        # torch.unique finds unique rows and returns the inverse indices
        # This effectively rebuilds the topology without a dictionary
        unique_verts, inverse_indices = torch.unique(flat_verts, dim=0, return_inverse=True)
        
        verts = unique_verts
        faces = inverse_indices.view(-1, 3)

    # Normalize to sphere
    verts = torch.nn.functional.normalize(verts, p=2, dim=1)
    
    return verts, faces