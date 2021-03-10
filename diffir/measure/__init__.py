from profane import ModuleBase, import_all_modules, ConfigOption


class Measure(ModuleBase):
    module_type = "measure"
    config_spec = [
        ConfigOption(key="metric", default_value="ndcg_20", description="The metric to use for selecting queries"),
        ConfigOption(key="topk", default_value=5, description="How many queries to retrieve"),
    ]

    # TODO finalize API
    def query_differences(self, run1, run2, *args, **kwargs):
        if run1 and run2:
            return self._query_differences(run1, run2, *args, **kwargs)
        elif run1 and run2 is None:
            return sorted(list(run1.keys()))[: self.config["topk"]]

    def _query_differences(self, run1, run2, *args, **kwargs):
        raise NotImplementedError


# TODO this is going to break once we introduce optional modules. need a way for them to fail gracefully.
#      or to enumerate/register them without importing the py file?
import_all_modules(__file__, __package__)
