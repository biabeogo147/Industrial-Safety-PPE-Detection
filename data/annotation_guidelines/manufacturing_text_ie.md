# Manufacturing Text IE Annotation Guideline

## Scope

This guideline defines how to annotate OSHA manufacturing sentences for the `BERT + BIEO` text IE task.

The annotation target is a set of relational triples per sentence.

## Predicate Set

- `be_equipped_with`
- `perform_operations`
- `occurrence`

## Triple Schema

- `worker -> be_equipped_with -> ppe`
- `worker -> perform_operations -> operation`
- `operation -> occurrence -> injury_or_hazard`

## Span Rules

- Use the shortest span that is semantically complete.
- Annotate the surface form as written in the sentence.
- Do not normalize the span text into canonical IDs during annotation.
- Include multi-word phrases such as `eye protection`, `respiratory protection`, and `welding operations`.

## Subject Rules

- When the sentence explicitly names the actor, annotate that actor span.
- Use the plural or singular form exactly as written, such as `employee`, `employees`, `worker`, or `workers`.
- Do not invent an implicit subject span if the sentence does not provide one.

## Object Rules

- For PPE relations, annotate the PPE phrase only.
- For operation relations, annotate the operation phrase only.
- For hazard relations, annotate the injury or hazard phrase only.

## Overlap Rules

- Preserve overlapping triples.
- If one subject participates in multiple relations, annotate every valid triple.
- If one sentence contains both PPE and operation context, annotate both relations.

## Exclusions

Do not annotate:

- navigation or website boilerplate
- title-only lines
- effective-date-only clauses
- Federal Register amendment history
- clauses that cannot support later image-grounded checking

## Example

Sentence:

`Employees exposed to flying particles shall use eye protection during grinding operations.`

Triples:

- `<Employees, be_equipped_with, eye protection>`
- `<Employees, perform_operations, grinding operations>`

## Quality Check

- Every span must fall within token boundaries.
- Every predicate must belong to the approved predicate set.
- Every sentence may have zero, one, or multiple triples.
