from typing import Callable, Hashable, Mapping, Optional, List, Iterable, Tuple, Dict

import pandas as pd


Artifact = Hashable
MapArtifacts = Mapping[str, List[Artifact]]
ExtractorArtifacts = Callable[[pd.Series], MapArtifacts]
ArtifactSelection = Optional[str]
SelectorArtifact = Callable[[str, str, Artifact], ArtifactSelection]


def as_is(name: str, variation: str, artifact: Artifact) -> ArtifactSelection:
    return name + variation


def ignore_variation(name: str, variation: str, artifact: Artifact) -> ArtifactSelection:
    return name


def drop(name: str, variation: str, artifact: Artifact) -> ArtifactSelection:
    return None


def extractor_map(
    names_selector: Mapping[str, SelectorArtifact],
    maybe_variations: Optional[List[str]] = None
) -> ExtractorArtifacts:
    if maybe_variations is None:
        variations = [""]
    else:
        variations = maybe_variations
        variations.insert(0, "")

    def extract_artifacts_using_map(row: pd.Series) -> MapArtifacts:
        map_artifacts: Dict[str, List[Artifact]] = {}
        for name, select in names_selector.items():
            for variation in variations:
                name_variated = name + variation
                if name_variated in row.index:
                    artifact = row[name_variated]
                    name_given = select(name, variation, artifact)
                    if name_given is not None:
                        map_artifacts.setdefault(name_given, []).append(artifact)
        return map_artifacts

    return extract_artifacts_using_map


def iter_map_artifacts(map_artifacts: MapArtifacts) -> Iterable[Tuple[str, Artifact]]:
    for name, artifacts in map_artifacts.items():
        for artifact in artifacts:
            yield (name, artifact)
