"""
TODO: Reduce complexity of data versions. Right now the full pipeline is (as I understand it):
1. Raw
2. PreMEDS
3. MEDS
4. Features
5. Tokenized

And then for PT:
6. Pretraining

Or for FT:
7. Cohort/finetune
8. outcomes
9. Finetuning

Or for test:
11. Cohort/held_out
12. Testing

I don't know exactly how they should be reduced but this seems like a very large amount of file versions and I think it could probably be reduced a bit
At least I am completely lost as to how to keep track of all these.

But i think the data preprocessing pipeline needs to cut at least a few of these intermediate steps out.

Things to add:

Save data as the torch tensors it will need to become:

Right now we do this in getitem:

        patient = self.patients[index]
        concepts = torch.tensor(patient.concepts, dtype=torch.long)
        masked_concepts, target = self.masker.mask_patient_concepts(concepts)
        attention_mask = torch.ones_like(masked_concepts)
        sample = {
            CONCEPT_FEAT: masked_concepts,
            TARGET: target,
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,

And I dont think there's any reason to not just save the items as the tensors they have to become, and potentially already
in the dict format its used in. Instead of the Patient dataclass.

"""
