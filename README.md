# Status
Skeleton has two types of modules:
- `measure` with possible values `qrel` and `topk` (e.g., `measure.name=topk measure.topk=5`)
- `weight` with possible value `exactmatch`

I created some placeholder APIs for these, but didn't think them through. I *think* profane's handling of modules will work well for making these pluggable, but it's hard to be certain at this stage and I could be convinced otherwise.

The CLI arguments are being parsed with profane's default approach, which mimics sacred.
I'm fine with modifying this later if people feel strongly about making it look more traditional.
Even if we stick with this syntax, we have fixed and known modules here, so `module.name=foo` could be replaced with just `module=foo`.

To run, try something like:
- `./run.sh run1 run2 with dataset=msmarco-passage queries=text measure.name=topk measure.topk=1`
- `./run.sh run1 run2 --cli with measure.topk=5`

