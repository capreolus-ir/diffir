from profane import ModuleBase, import_all_modules


class Weight(ModuleBase):
    module_type = "weight"

    # TODO finalize API
    def score_document_regions(self, query, doc, run_idx):
        raise NotImplementedError()


# TODO this is going to break once we introduce optional modules. need a way for them to fail gracefully.
#      or to enumerate/register them without importing the py file?
import_all_modules(__file__, __package__)
