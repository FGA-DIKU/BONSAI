from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from bonsai.paths import get_config_path


class DataCreationSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="data-generation-searchpath-plugin",
            path="file://" + get_config_path() + "/data_creation",
        )


class TestingSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="testing-searchpath-plugin",
            path="file://" + get_config_path() + "/tests",
        )
