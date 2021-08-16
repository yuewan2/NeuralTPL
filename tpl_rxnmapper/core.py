"""Core RXN Attention Mapper module."""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import re
import os
import pkg_resources
import torch
import numpy as np
from rdkit import Chem

from typing import Optional, List, Dict, Tuple, Union
from transformers import PreTrainedModel, AlbertModel, BertModel, RobertaModel
from .tokenization_smiles import SmilesTokenizer
from .attention import AttentionScorer
from .smiles_utils import generate_atom_mapped_reaction_atoms, process_reaction

MODEL_TYPE_DICT = {
    "bert": BertModel,
    "albert": AlbertModel,
    "roberta": RobertaModel
}

LOGGER = logging.getLogger("rxnmapper:core")

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|<MASK>|<unk>|>>|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|≈|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    assert smi == ''.join(tokens)
    return tokens

class RXNMapper:
    """ Wrap the Transformer model, corresponding tokenizer, and attention scoring algorithms.


    Maps product atoms to reactant atoms using the attention weights 
    """

    def __init__(
        self,
        config: Dict = {},
        logger: Optional[logging.Logger] = None,
    ):
        """
        RXNMapper constructor.

        Args:
            config (Dict): Config dict, leave it empty to have the
                official rxnmapper.
            logger (logging.Logger, optional): a logger.
                Defaults to None, a.k.a using a default logger.
            
            Example:
            
            >>> from rxnmapper import RXNMapper
            >>> rxn_mapper = RXNMapper()
        """

        # Config takes "model_path", "model_type", "attention_multiplier", "head", "layers"
        self.model_path = config.get(
            "model_path",
            pkg_resources.resource_filename(
                "tpl_rxnmapper",
                "models/transformers/albert_heads_8_uspto_all_1310k"
            ),
        )
        self.model_type = config.get("model_type", "albert")
        self.attention_multiplier = config.get("attention_multiplier", 90.0)
        self.head = config.get("head", 5)
        self.layers = config.get("layers", [10])

        self.logger = logger if logger else LOGGER
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _load_model_and_tokenizer(self) -> Tuple:
        """
        Load transformer and tokenizer model.

        Returns:
            Tuple: containing model and tokenizer
        """
        model_class = MODEL_TYPE_DICT[self.model_type]
        model = model_class.from_pretrained(
            self.model_path,
            output_attentions=True,
            output_past=False,
            output_hidden_states=False,
        )

        vocab_path = None

        if os.path.exists(os.path.join(self.model_path, "vocab.txt")):
            vocab_path = os.path.join(self.model_path, "vocab.txt")

        tokenizer = SmilesTokenizer(
            vocab_path, max_len=model.config.max_position_embeddings
        )
        return (model, tokenizer)

    def convert_batch_to_attns(
        self,
        rxn_smiles_list: List[str],
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
    ):
        """ Extract desired attentions from a given batch of reactions.

        Args:
            rxn_smiles_list: List of reactions to mape
            force_layer: If given, override the default layer used for RXNMapper
            force_head: If given, override the default head used for RXNMapper
        """
        if force_layer is None: use_layers = self.layers
        else: use_layers = [force_layer]

        if force_head is None: use_head = self.head
        else: use_head = force_head

        encoded_ids = self.tokenizer.batch_encode_plus(
            rxn_smiles_list,
            padding=True,
            return_tensors="pt",
        )
        parsed_input = { 
            k: v.to(self.device) for k, v in encoded_ids.items()
        }
        with torch.no_grad():
            output = self.model(**parsed_input)
        attentions = output[2]
        selected_attns = torch.cat(
            [
                a.unsqueeze(1)
                for i, a in enumerate(attentions) if i in use_layers
            ],
            dim=1,
        )

        selected_attns = selected_attns[:, :, use_head, :, :]
        selected_attns = torch.mean(selected_attns, dim=[1])
        att_masks = encoded_ids["attention_mask"].to(torch.bool)

        selected_attns = [
            a[mask][:, mask] for a, mask in zip(selected_attns, att_masks)
        ]
        return selected_attns

    def tokenize_for_model(self, rxn: str):
        """Tokenize a reaction SMILES with the special tokens needed for the model"""
        return (
            [self.tokenizer.cls_token] +
            self.tokenizer.basic_tokenizer.tokenize(rxn) +
            [self.tokenizer.sep_token]
        )

    def get_attention_guided_atom_maps(
        self,
        rxns: List[str],
        zero_set_p: bool = True,
        zero_set_r: bool = True,
        canonicalize_rxns: bool = True,
        detailed_output: bool = False,
        absolute_product_inds: bool = False,
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
        sanitize_token = True
    ):
        """Generate atom-mapping for reactions.

        Args:
            rxns: List of reaction SMILES (no reactant/reagent split)
            zero_set_p: Mask mapped product atoms (default: True)
            zero_set_r: Mask mapped reactant atoms (default: True)
            canonicalize_rxns: Canonicalize reactions (default: True)
            detailed_output: Get back more information (default: False)
            absolute_product_inds: Different atom indexing (default: False)
            force_layer: Force specific layer (default: None)
            force_head: Force specific head (default: None)

        Returns:
            Mapped reactions with confidence score (List):
                - mapped_rxn: Mapped reaction SMARTS
                - confidence: Model confidence in the mapping rxn

            `detailed_output=True` additionally outputs...
                - pxr_mapping_vector: Vector used to generate the product atom indexes for the mapping
                - pxr_confidences: The corresponding confidence for each atom's map
                - mapping_tuples: (product_atom_index (relative to first product atom), corresponding_reactant_atom, confidence)
                - pxrrxp_attns: Just the attentions from the product tokens to the reactant tokens
                - tokensxtokens_attns: Full attentions for all tokens
                - tokens: Tokens that were inputted into the model
        """
        results = []

        if canonicalize_rxns:
            rxns = [process_reaction(rxn) for rxn in rxns]

        def convert_token(rxn):
            additional_dictionary = {'[#14]': '[Si]', '[#15;a]': 'p', '[#15]': 'P', '[#16;a]': 's', '[#16]': 'S',
                                     '[#34;a]': '[se]', '[#34]': '[Se]', '[#35]': 'Br', '[#50]': '[Sn]', '[#53]': 'I',
                                     '[#5]': 'B', '[#7;+]': '[N+]', '[#7;-]': '[N-]', '[#7;a;+]': '[n+]', '[#7;a]': 'n',
                                     '[#7]': 'N', '[#8;-]': '[O-]', '[#8;a]': 'o', '[#8]': 'O', '[#9]': 'F',
                                     '[BrH]': '[BrH+]', '[ClH]':'[ClH+]', '[NH3]':'[NH3+]', '[N]':'[N]',
                                     '[O]':'[O]', '[cH]':'[c]', '[n]':'[n]', '[o]':'[o]'}
            rxn_token, rxn_token_new = smi_tokenizer(rxn), []
            for token in rxn_token:
                if token in additional_dictionary:
                    rxn_token_new.append(additional_dictionary[token])
                else:
                    rxn_token_new.append(token)

            new_rxn = ''.join(rxn_token_new)
            return new_rxn


        batch_rxns = [convert_token(rxn) for rxn in rxns]
        if sanitize_token:
            attns = self.convert_batch_to_attns(
                batch_rxns, force_layer=force_layer, force_head=force_head
            )
        else:
            attns = self.convert_batch_to_attns(
                rxns, force_layer=force_layer, force_head=force_head
            )

        for attn, rxn, brxn in zip(attns, rxns, batch_rxns):
            just_tokens = self.tokenize_for_model(rxn)
            tokensxtokens_attn = attn.detach().cpu().numpy()
            attention_scorer = AttentionScorer(
                rxn,
                just_tokens,
                tokensxtokens_attn,
                attention_multiplier=self.attention_multiplier,
                mask_mapped_product_atoms=zero_set_p,
                mask_mapped_reactant_atoms=zero_set_r,
                output_attentions=
                detailed_output,  # Return attentions when detailed output requested
            )

            output = attention_scorer.generate_attention_guided_pxr_atom_mapping(
                absolute_product_inds=absolute_product_inds
            )

            result = {
                "mapped_rxn":
                    generate_atom_mapped_reaction_atoms(
                        rxn, output["pxr_mapping_vector"]
                    ),
                "confidence":
                    np.prod(output["confidences"]),
            }
            if detailed_output:
                result["pxr_mapping_vector"] = output["pxr_mapping_vector"]
                result["pxr_confidences"] = output["confidences"]
                result["mapping_tuples"] = output["mapping_tuples"]
                result["pxrrxp_attns"] = output["pxrrxp_attns"]
                result["tokensxtokens_attns"] = tokensxtokens_attn
                result["tokens"] = just_tokens

            results.append(result)
        return results