"""
Unit tests for import functionality across all TASNI modules
"""

import pytest


class TestCoreImports:
    """Test core module imports"""

    def test_import_config(self):
        """Test config module import"""
        from tasni.core import config

        assert config is not None

    def test_import_config_env(self):
        """Test config_env module import"""
        from tasni.core import config_env

        assert config_env is not None

    def test_import_tasni_logging(self):
        """Test tasni_logging module import"""
        from tasni.core import tasni_logging

        assert tasni_logging is not None


class TestDownloadImports:
    """Test download module imports"""

    def test_import_async_neowise_query(self):
        """Test async_neowise_query import"""
        try:
            from tasni.download import async_neowise_query

            assert async_neowise_query is not None
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_download_wise_full(self):
        """Test download_wise_full import"""
        from tasni.download import download_wise_full

        assert download_wise_full is not None

    def test_import_download_gaia_full(self):
        """Test download_gaia_full import"""
        from tasni.download import download_gaia_full

        assert download_gaia_full is not None


class TestCrossmatchImports:
    """Test crossmatch module imports"""

    def test_import_crossmatch_full(self):
        """Test crossmatch_full import"""
        from tasni.crossmatch import crossmatch_full

        assert crossmatch_full is not None

    def test_import_gpu_crossmatch(self):
        """Test gpu_crossmatch import (optional)"""
        try:
            from tasni.crossmatch import gpu_crossmatch

            assert gpu_crossmatch is not None
        except ImportError as e:
            pytest.skip(f"GPU not available: {e}")

    def test_import_optimized_crossmatch(self):
        """Test optimized_crossmatch import"""
        from tasni.optimized import optimized_crossmatch

        assert optimized_crossmatch is not None


class TestAnalysisImports:
    """Test analysis module imports"""

    def test_import_analyze_kinematics(self):
        """Test analyze_kinematics import"""
        from tasni.analysis import analyze_kinematics

        assert analyze_kinematics is not None

    def test_import_compute_ir_variability(self):
        """Test compute_ir_variability import"""
        from tasni.analysis import compute_ir_variability

        assert compute_ir_variability is not None

    def test_import_periodogram_analysis(self):
        """Test periodogram_analysis import"""
        from tasni.analysis import periodogram_analysis

        assert periodogram_analysis is not None


class TestFilteringImports:
    """Test filtering module imports"""

    def test_import_filter_anomalies_full(self):
        """Test filter_anomalies_full import"""
        from tasni.filtering import filter_anomalies_full

        assert filter_anomalies_full is not None

    def test_import_multi_wavelength_scoring(self):
        """Test multi_wavelength_scoring import"""
        from tasni.filtering import multi_wavelength_scoring

        assert multi_wavelength_scoring is not None

    def test_import_validate_brown_dwarfs(self):
        """Test validate_brown_dwarfs import"""
        from tasni.filtering import validate_brown_dwarfs

        assert validate_brown_dwarfs is not None


class TestGenerationImports:
    """Test generation module imports"""

    def test_import_generate_golden_list(self):
        """Test generate_golden_list import"""
        from tasni.generation import generate_golden_list

        assert generate_golden_list is not None

    def test_import_generate_publication_figures(self):
        """Test generate_publication_figures import"""
        from tasni.generation import generate_publication_figures

        assert generate_publication_figures is not None

    def test_import_prepare_spectroscopy_targets(self):
        """Test prepare_spectroscopy_targets import"""
        from tasni.generation import prepare_spectroscopy_targets

        assert prepare_spectroscopy_targets is not None


class TestMLImports:
    """Test ML module imports (optional)"""

    def test_import_xpu_classify(self):
        """Test xpu_classify import (optional)"""
        try:
            from tasni.ml import xpu_classify

            assert xpu_classify is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Intel XPU not available: {e}")

    def test_import_cluster_images_xpu(self):
        """Test cluster_images_xpu import (optional)"""
        try:
            from tasni.ml import cluster_images_xpu

            assert cluster_images_xpu is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Intel XPU not available: {e}")


class TestUtilsImports:
    """Test utils module imports"""

    def test_import_quick_check(self):
        """Test quick_check import"""
        from tasni.utils import quick_check

        assert quick_check is not None

    def test_import_fast_cutouts(self):
        """Test fast_cutouts import"""
        from tasni.utils import fast_cutouts

        assert fast_cutouts is not None


class TestChecksImports:
    """Test checks module imports"""

    def test_import_check_spectra(self):
        """Test check_spectra import"""
        try:
            from tasni.checks import check_spectra

            assert check_spectra is not None
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_check_spitzer(self):
        """Test check_spitzer import"""
        from tasni.checks import check_spitzer

        assert check_spitzer is not None

    def test_import_check_tess(self):
        """Test check_tess import"""
        from tasni.checks import check_tess

        assert check_tess is not None


class TestOptimizedImports:
    """Test optimized module imports"""

    def test_import_optimized_pipeline(self):
        """Test optimized_pipeline import"""
        from tasni.optimized import optimized_pipeline

        assert optimized_pipeline is not None

    def test_import_optimized_variability(self):
        """Test optimized_variability import"""
        from tasni.optimized import optimized_variability

        assert optimized_variability is not None
