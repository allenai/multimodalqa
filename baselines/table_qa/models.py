from transformers.modeling_bert import *
from transformers.modeling_roberta import *
from data_reader import ans_type2id_map


@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`), and an answer type 
    classification head on top for classifying `span_answer`, `yes`, `no`. """,
    BERT_START_DOCSTRING,
)
class BertQaWithYesNoHead(BertForQuestionAnswering):
    def __init__(self, config):
        super(BertQaWithYesNoHead, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.span_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.ans_type_outputs = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        answer_types=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        span_logits = self.span_outputs(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        answer_type_logits = self.ans_type_outputs(pooled_output)

        outputs = (start_logits, end_logits, answer_type_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None and answer_types is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(answer_types.size()) > 1:
                answer_types = answer_types.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2
            span_loss *= (answer_types == 0).float()
            type_loss = loss_fct(answer_type_logits, answer_types)
            total_loss = (span_loss + type_loss).mean()

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, answer_type_logits, (hidden_states), (attentions)


class BertQaForTable(BertForQuestionAnswering):
    def __init__(self, config):
        super(BertQaForTable, self).__init__(config)
        self.token_outputs = nn.Linear(config.hidden_size, 1)
        self.ans_type_outputs = nn.Linear(config.hidden_size, len(ans_type2id_map))
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        cell_token_masks=None,
        cell_labels=None,
        answer_types=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        token_logits = self.token_outputs(sequence_output)
        token_logits_for_cells = torch.transpose(token_logits, 1, 2) * cell_token_masks
        token_num_for_cells = torch.sum(cell_token_masks, 2)
        cell_masks = (token_num_for_cells > 0).float()
        # use one to avoid dividing by 0,
        # and the final logits should be 0 since all the logits are masked for empty cells
        token_num_for_cells = torch.where(
            token_num_for_cells > 0,
            token_num_for_cells,
            torch.ones_like(token_num_for_cells)
        )
        cell_logits = torch.sum(token_logits_for_cells, 2) / token_num_for_cells
        cell_probs = torch.sigmoid(cell_logits) * cell_masks

        # cell_logits = torch.masked_fill(cell_logits, ~(token_num_for_cells > 0), -1e10)
        # cell_probs = torch.softmax(cell_logits, dim=1)

        ans_type_logits = self.ans_type_outputs(pooled_output)
        outputs = (cell_probs, ans_type_logits) + outputs[2:]
        # outputs = (cell_logits, ans_type_logits) + outputs[2:]

        if cell_labels is not None and answer_types is not None:
            # If we are on multi-GPU, split add a dimension
            if len(cell_labels.size()) > 2:
                cell_labels = cell_labels.squeeze(-1)
            if len(answer_types.size()) > 1:
                answer_types = answer_types.squeeze(-1)

            # BCE loss for each cell
            cell_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                cell_logits, cell_labels.float(), reduction="none"
            )
            cell_losses = cell_losses * cell_masks
            cell_loss = torch.sum(cell_losses, 1) / torch.sum(cell_masks, 1)

            # negative log marginal likelihood loss for all the target cells
            # target_cell_prob_sum = torch.sum(cell_probs * cell_labels, 1)
            # cell_loss = - torch.log(target_cell_prob_sum)

            # soft cross entropy loss
            # gold_distribution = cell_labels.float() / torch.sum(cell_labels, 1, keepdim=True)
            # cell_loss = - torch.sum(gold_distribution * torch.log(cell_probs))

            type_loss = torch.nn.functional.cross_entropy(ans_type_logits, answer_types, reduction="none")
            total_loss = (cell_loss + type_loss).mean()
            outputs = (total_loss,) + outputs

        return outputs  # (loss), cell_probs, answer_type_logits, (hidden_states), (attentions)


class RobertaQaForTable(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(RobertaQaForTable, self).__init__(config)
        self.token_outputs = nn.Linear(config.hidden_size, 1)
        self.ans_type_outputs = nn.Linear(config.hidden_size, len(ans_type2id_map))
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        cell_token_masks=None,
        cell_labels=None,
        answer_types=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        token_logits = self.token_outputs(sequence_output)
        token_logits_for_cells = torch.transpose(token_logits, 1, 2) * cell_token_masks
        token_num_for_cells = torch.sum(cell_token_masks, 2)
        cell_masks = (token_num_for_cells > 0).float()
        # use one to avoid dividing by 0,
        # and the final logits should be 0 since all the logits are masked for empty cells
        token_num_for_cells = torch.where(
            token_num_for_cells > 0,
            token_num_for_cells,
            torch.ones_like(token_num_for_cells)
        )
        cell_logits = torch.sum(token_logits_for_cells, 2) / token_num_for_cells
        cell_probs = torch.sigmoid(cell_logits) * cell_masks

        # cell_logits = torch.masked_fill(cell_logits, ~(token_num_for_cells > 0), -1e10)
        # cell_probs = torch.softmax(cell_logits, dim=1)

        ans_type_logits = self.ans_type_outputs(pooled_output)
        outputs = (cell_probs, ans_type_logits) + outputs[2:]
        # outputs = (cell_logits, ans_type_logits) + outputs[2:]

        if cell_labels is not None and answer_types is not None:
            # If we are on multi-GPU, split add a dimension
            if len(cell_labels.size()) > 2:
                cell_labels = cell_labels.squeeze(-1)
            if len(answer_types.size()) > 1:
                answer_types = answer_types.squeeze(-1)

            # BCE loss for each cell
            cell_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                cell_logits, cell_labels.float(), reduction="none"
            )
            cell_losses = cell_losses * cell_masks
            cell_loss = torch.sum(cell_losses, 1) / torch.sum(cell_masks, 1)

            # negative log marginal likelihood loss for all the target cells
            # target_cell_prob_sum = torch.sum(cell_probs * cell_labels, 1)
            # cell_loss = - torch.log(target_cell_prob_sum)

            # soft cross entropy loss
            # gold_distribution = cell_labels.float() / torch.sum(cell_labels, 1, keepdim=True)
            # cell_loss = - torch.sum(gold_distribution * torch.log(cell_probs))

            type_loss = torch.nn.functional.cross_entropy(ans_type_logits, answer_types, reduction="none")
            total_loss = (cell_loss + type_loss).mean()
            outputs = (total_loss,) + outputs

        return outputs  # (loss), cell_probs, answer_type_logits, (hidden_states), (attentions)