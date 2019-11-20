import numpy as np


def export_to_tf_projector(output_file, embeddings):
    np.savetxt(output_file,
               embeddings,
               delimiter="\t")
