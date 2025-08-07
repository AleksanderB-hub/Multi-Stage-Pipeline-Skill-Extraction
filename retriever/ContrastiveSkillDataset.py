from typing import List, Dict, Set, Tuple, Optional, Any, Union
import random
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

@dataclass
class SkillData:
    """Data structure for individual skill information"""
    skill_label: str
    skill_class: str
    examples: List[str]
    esco_category: Optional[str] = None  # Higher-level ESCO category
    
class ContrastiveSkillDataset(Dataset):
    """This is a Dataset class for contrastive learning on skill classification.
    It handles data augmentation, hard negative sampling strategies, and ESCO-based sampling.

    Args:
        Dataset: The Json (list of dictionaries) containing all of the data: skill_labels, examples, definitions, ESCO categories. 
    """
    
    def __init__(
        self,
        args,
        skill_data: List[Dict],
        allow_pre_train
    ):
        self.augmentation_ratio = args.augmentation_ratio
        self.hard_negative_strategy = args.hard_negative_strategy
        self.num_hard_negatives = args.num_hard_negatives
        self.enable_augmentation = args.enable_augmentation
        # set the random seed
        self.rng = random.Random(args.seed)
        # determine the stage
        self.pre_train = allow_pre_train
        # prepare the data
        self.skills = self._process_raw_data(skill_data)
        self.training_examples = self._create_training_examples()
        # initiate the similarity matrix (to be computed later with FAISS)
        self.similarity_matrix = None
        
        
    def _process_raw_data(self, raw_data: List[Dict]) -> List[SkillData]:
        """Convert dictionary data into SkillData objects

        Args:
            raw_data (List[Dict]): original list of dictionaries

        Returns:
            List[SkillData]: The new data
        """
        
        skills = []
        item_to_chose = 'skill_label'
        
        for item in raw_data:
            skill = SkillData(
                skill_label=item['skill_label'],
                skill_class=item[f'{item_to_chose}'],
                examples=item['examples'],
                esco_category=item.get("ESCO_category", None)
            )
            skills.append(skill)
            
        return skills
    

    def _create_training_examples(self) -> List[Tuple]:
        """Create Training examples from skill data

        Returns:
            List[Tuple]: The training data organised in the expected format
        """

        examples = []

        # Calculate total number of examples for progress bar
        total_examples = sum(len(skill.examples) for skill in self.skills)

        # Create progress bar
        from tqdm import tqdm
        pbar = tqdm(total=total_examples, desc="Creating training examples", unit="examples")

        for skill_idx, skill in enumerate(self.skills):
            for example_sentence in skill.examples:
                # Update progress bar
                pbar.update(1)

                # The augmentation operation
                should_augment = (
                    self.enable_augmentation and self.rng.random() < self.augmentation_ratio
                )

                # Augment sentences based on the stage
                if should_augment:
                    if self.pre_train:
                        augmented_result = self._create_augmented_sentence_pre_train(skill.skill_label, skill_idx)
                    else:
                        augmented_result = self._create_augmented_sentence(example_sentence, skill_idx)
                        
                    # Check whether the sentence was augmented correctly
                    if augmented_result:
                        augmented_sentence, excluded_skill_idx = augmented_result
                        examples.append((
                            augmented_sentence,  # new anchor
                            example_sentence if self.pre_train else skill.skill_class, # either definition or label from examples depending on the stage
                            skill_idx,          # skill index for negative sampling
                            excluded_skill_idx  # the index of skill of augmented sentence 
                        )) 
                    # If augmentation failed
                    else:
                        examples.append((
                            skill.skill_label if self.pre_train else example_sentence,
                            example_sentence if self.pre_train else skill.skill_class,
                            skill_idx,
                            None
                        ))
                else:
                    examples.append((
                        skill.skill_label if self.pre_train else example_sentence,
                        example_sentence if self.pre_train else skill.skill_class,
                        skill_idx,
                        None
                    ))
                    
        pbar.close()

        return examples
    def _create_augmented_sentence(self, primary_sentence: str, primary_skill_idx: int) -> Optional[Tuple[str, int]]:
        """Function to create augmented anchors     

        Args:
            primary_sentence (str): the original (main) sentence
            primary_skill_idx (int): the index of the sentence for reference    

        Returns:
            Optional[Tuple[str, int]]: the augmented sentence and index to exclude 
        """
        # Get list of all skill indices except the primary one
        available_skills = list(range(len(self.skills)))
        available_skills.remove(primary_skill_idx)
        
        # Shuffle to randomize which skills we try
        self.rng.shuffle(available_skills)
        
        # Try multiple times to find a suitable augmentation partner
        max_attempts = min(20, len(available_skills))  # Don't try more than available skills
        
        for attempt in range(max_attempts):
            # Pick a random secondary skill
            secondary_skill_idx = available_skills[attempt]
            secondary_skill = self.skills[secondary_skill_idx]
            
            # Skip if this skill has no examples
            if not secondary_skill.examples:
                continue
            
            # Pick a random example from the secondary skill
            secondary_sentence = self.rng.choice(secondary_skill.examples)
        
            # Randomly decide the order
            if self.rng.random() < 0.5:
                augmented_sentence = f"{primary_sentence} {secondary_sentence}"
            else:
                augmented_sentence = f"{secondary_sentence} {primary_sentence}"
            
            return augmented_sentence, secondary_skill_idx
        
        # If we couldn't find a suitable partner after max_attempts
        return None
    
    def _create_augmented_sentence_pre_train(self, primary_definition: str, primary_skill_idx: int) -> Optional[Tuple[str, int]]:
        """Function to create augmented definitions for pre-training

        Args:
            primary_definition (str): the original skill definition
            primary_skill_idx (int): the index of the primary skill for reference    

        Returns:
            Optional[Tuple[str, int]]: the augmented definition and secondary skill index to exclude 
        """
        # Get list of all skill indices except the primary one
        available_skills = list(range(len(self.skills)))
        available_skills.remove(primary_skill_idx)

        # Shuffle to randomize which skills we try
        self.rng.shuffle(available_skills)

        # Try multiple times to find a suitable augmentation partner
        max_attempts = min(20, len(available_skills))

        for attempt in range(max_attempts):
            # Pick a random secondary skill
            secondary_skill_idx = available_skills[attempt]
            secondary_skill = self.skills[secondary_skill_idx]

            # Get the secondary definition (stored in skill_label for pre-training data)
            secondary_definition = secondary_skill.skill_label

            # Skip if secondary definition is empty
            if not secondary_definition or not secondary_definition.strip():
                continue

            # Randomly decide the order
            if self.rng.random() < 0.5:
                augmented_definition = f"{primary_definition} {secondary_definition}"
            else:
                augmented_definition = f"{secondary_definition} {primary_definition}"

            return augmented_definition, secondary_skill_idx

        # If we couldn't find a suitable partner after max_attempts
        return None
    
    def _get_esco_negatives_with_indices(
        self, 
        positive_skill_idx: int, 
        excluded_skill_idx: Optional[int] = None,
        num: Optional[int] = None
    ) -> Tuple[List[str], List[int]]:
        """Get ESCO negatives with their skill indices"""

        if num is None:
            num = self.num_hard_negatives

        positive_skill = self.skills[positive_skill_idx]

        if not positive_skill.esco_category:
            return self._get_random_negatives_with_indices(
                positive_skill_idx, num, excluded_skill_idx
            )

        esco_negatives = []
        esco_indices = []

        # retrieve all suitable esco negatives
        for idx, skill in enumerate(self.skills):
            if (idx != positive_skill_idx and 
                (excluded_skill_idx is None or idx != excluded_skill_idx) and
                skill.esco_category == positive_skill.esco_category):
                esco_negatives.append(skill.definition)
                esco_indices.append(idx)

        # select esco negatives
        if len(esco_negatives) > num:
            combined = list(zip(esco_negatives, esco_indices))
            self.rng.shuffle(combined)
            esco_negatives, esco_indices = zip(*combined[:num])
            esco_negatives, esco_indices = list(esco_negatives), list(esco_indices)

        # Supplement with random ones if no sufficient esco were found
        if len(esco_negatives) < num:
            additional_needed = num - len(esco_negatives)
            rand_negs, rand_indices = self._get_random_negatives_with_indices(
                positive_skill_idx, additional_needed, excluded_skill_idx,
                # safeguard for already sourced esco negatives
                exclude_indices=set(esco_indices)  
            )
            esco_negatives.extend(rand_negs)
            esco_indices.extend(rand_indices)

        return esco_negatives[:num], esco_indices[:num] 

    def _get_random_negatives_with_indices(
        self, 
        positive_skill_idx: int, 
        num_negatives: int,
        excluded_skill_idx: Optional[int] = None,
        exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Get random negatives with their skill indices

        Args:
            positive_skill_idx (int): the index of the skill associated with the primary skill sentence
            num_negatives (int, optional): the total number of negatives to be retrieved. Defaults to None.
            excluded_skill_idx (int, optional): the indices of skills associated with secondary (augmented) sentence
            exclude_indices (int, optional): the list of indices to exclude 
        Returns:
            List[str]: The list of suitable negatives 
        """
        if exclude_indices is None:
            exclude_indices = set()

        # Build available indices
        available_indices = [
            i for i in range(len(self.skills))
            if i != positive_skill_idx and 
            (excluded_skill_idx is None or i != excluded_skill_idx) and
            i not in exclude_indices
        ]

        selected_indices = self.rng.sample(
            available_indices,
            min(num_negatives, len(available_indices))
        )

        negatives = [self.skills[idx].definition for idx in selected_indices]

        return negatives, selected_indices

    def __len__(self):
        return len(self.training_examples)
    
    def __getitem__(self, idx):

        anchor_text, positive_definition, skill_idx, excluded_skill_idx = self.training_examples[idx]
        negatives = []
        negative_skill_indices = []

        if self.hard_negative_strategy == 'esco':
            negatives, negative_skill_indices = self._get_esco_negatives_with_indices(
                skill_idx, excluded_skill_idx
            )

        elif self.hard_negative_strategy == 'in_batch':
            negatives, negative_skill_indices = [], []

        elif self.hard_negative_strategy == "mixed_batch_esco":
            # Only ESCO negatives (in-batch added by collator)
            negatives, negative_skill_indices = self._get_esco_negatives_with_indices(
                skill_idx, excluded_skill_idx, num=self.num_hard_negatives//2
            )
            
        else:
            raise ValueError(f"Unknown hard_negative_strategy: {self.hard_negative_strategy}")

        return {
            "anchor": anchor_text,
            "positive": positive_definition,
            "negatives": negatives,
            "negative_skill_indices": negative_skill_indices,
            "skill_idx": skill_idx,
            "excluded_skill_idx": excluded_skill_idx,
            "strategy": self.hard_negative_strategy,
            "is_augmented": excluded_skill_idx is not None
        }