import { useEffect, useMemo, useState } from "react";

import {
	getBaseInspectionColumnKeys,
	toSafeString,
} from "@/features/inspections/components/inspections-panel.utils";
import { INSPECTION_COLUMNS } from "@/features/inspections/data";
import type {
	InspectionColumnKey,
	InspectionRow,
	InspectionViewId,
} from "@/features/inspections/types";
import { getFloatingPanelAnchor } from "@/shared/utils/floating-panel";
import {
	createAdvancedDateRangeFilterToken,
	getAdvancedDateRangeFromSelectedValues,
	hasActiveTableFilters,
	isAdvancedDateRangeFilterToken,
	matchesAdvancedFilterCellValue,
	splitAdvancedFilterCellValue,
} from "@/shared/utils/table-filters";

type SortDirection = "asc" | "desc";

type UseInspectionsTableStateArgs = {
	inspectionRows: InspectionRow[];
	tableViewStorageKey?: string;
	tableViewStorageArea?: "localStorage" | "sessionStorage";
};

type HiddenColumnsByView = Record<InspectionViewId, InspectionColumnKey[]>;

const DEFAULT_INSPECTIONS_PAGE_SIZE = 25;

function getResponsiveInspectionsPageSize(viewportHeight: number) {
	if (viewportHeight < 860) {
		return 19;
	}

	if (viewportHeight < 1080) {
		return 25;
	}

	return 30;
}

function createRange(start: number, end: number) {
	const length = end - start + 1;
	return Array.from({ length }, (_, index) => index + start);
}

type PaginationItem = number | "ellipsis";

function buildPaginationItems(
	currentPage: number,
	totalPages: number,
	siblingCount = 1,
): PaginationItem[] {
	const totalPageNumbers = siblingCount + 5;

	if (totalPageNumbers >= totalPages) {
		return createRange(1, totalPages);
	}

	const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
	const rightSiblingIndex = Math.min(currentPage + siblingCount, totalPages);

	const shouldShowLeftDots = leftSiblingIndex > 2;
	const shouldShowRightDots = rightSiblingIndex < totalPages - 2;

	if (!shouldShowLeftDots && shouldShowRightDots) {
		const leftItemCount = 3 + 2 * siblingCount;
		const leftRange = createRange(1, leftItemCount);
		return [...leftRange, "ellipsis", totalPages];
	}

	if (shouldShowLeftDots && !shouldShowRightDots) {
		const rightItemCount = 3 + 2 * siblingCount;
		const rightRange = createRange(totalPages - rightItemCount + 1, totalPages);
		return [1, "ellipsis", ...rightRange];
	}

	if (shouldShowLeftDots && shouldShowRightDots) {
		const middleRange = createRange(leftSiblingIndex, rightSiblingIndex);
		return [1, "ellipsis", ...middleRange, "ellipsis", totalPages];
	}

	return createRange(1, totalPages);
}

function getAllInspectionColumnKeys() {
	return INSPECTION_COLUMNS.map((column) => column.key);
}

function getDefaultHiddenColumnsForView(viewId: InspectionViewId) {
	if (viewId === "calosc") {
		return [] as InspectionColumnKey[];
	}

	const baseColumns = new Set(getBaseInspectionColumnKeys(viewId));
	return getAllInspectionColumnKeys().filter(
		(columnKey) => !baseColumns.has(columnKey),
	);
}

function createDefaultHiddenColumnsByView(): HiddenColumnsByView {
	return {
		calosc: getDefaultHiddenColumnsForView("calosc"),
		podstawowe: getDefaultHiddenColumnsForView("podstawowe"),
		terminy: getDefaultHiddenColumnsForView("terminy"),
	};
}

function splitInspectionAdvancedFilterCellValue(
	columnKey: InspectionColumnKey,
	rawValue: string,
) {
	if (columnKey !== "zakresInspekcji") {
		return splitAdvancedFilterCellValue(rawValue);
	}

	const normalizedValue = rawValue.trim();
	if (!normalizedValue) {
		return ["(puste)"];
	}

	const tokens = normalizedValue
		.split(/[;\n]/)
		.map((item) => item.trim())
		.filter(Boolean);

	if (tokens.length === 0) {
		return ["(puste)"];
	}

	return Array.from(new Set(tokens));
}

function matchesInspectionAdvancedFilterCellValue(
	columnKey: InspectionColumnKey,
	rawValue: string,
	selectedValues: string[],
) {
	if (columnKey !== "zakresInspekcji") {
		return matchesAdvancedFilterCellValue(rawValue, selectedValues);
	}

	if (selectedValues.length === 0) {
		return true;
	}

	const selectedDateRange = getAdvancedDateRangeFromSelectedValues(selectedValues);
	const selectedDiscreteValues = selectedValues.filter(
		(value) => !isAdvancedDateRangeFilterToken(value),
	);

	const cellTokens = splitInspectionAdvancedFilterCellValue(columnKey, rawValue);
	const hasDiscreteMatch =
		selectedDiscreteValues.length === 0 ||
		cellTokens.some((token) => selectedDiscreteValues.includes(token));

	if (!hasDiscreteMatch) {
		return false;
	}

	if (selectedDateRange) {
		return false;
	}

	return true;
}

export function useInspectionsTableState({
	inspectionRows,
	tableViewStorageKey,
	tableViewStorageArea = "sessionStorage",
}: UseInspectionsTableStateArgs) {
	const [hiddenColumnsByView, setHiddenColumnsByView] =
		useState<HiddenColumnsByView>(() => createDefaultHiddenColumnsByView());
	const [hiddenColumns, setHiddenColumns] = useState<InspectionColumnKey[]>(
		() => getDefaultHiddenColumnsForView("calosc"),
	);
	const [selectedInspectionView, setSelectedInspectionView] =
		useState<InspectionViewId>("calosc");
	const [isColumnPickerOpen, setIsColumnPickerOpen] = useState(false);
	const [isAdvancedFilterModalOpen, setIsAdvancedFilterModalOpen] =
		useState(false);
	const [advancedFilterColumnKey, setAdvancedFilterColumnKey] =
		useState<InspectionColumnKey>("nazwaPodmiotu");
	const [advancedFilterSearch, setAdvancedFilterSearch] = useState("");
	const [advancedFilterAnchor, setAdvancedFilterAnchor] = useState({
		top: 120,
		left: 120,
	});
	const [advancedFilters, setAdvancedFilters] = useState<
		Partial<Record<InspectionColumnKey, string[]>>
	>({});
	const [draftSelectedInspectionView, setDraftSelectedInspectionView] =
		useState<InspectionViewId>("calosc");
	const [draftHiddenColumns, setDraftHiddenColumns] = useState<
		InspectionColumnKey[]
	>([]);
	const [draftHiddenColumnsByView, setDraftHiddenColumnsByView] =
		useState<HiddenColumnsByView>(() => createDefaultHiddenColumnsByView());
	const [sortColumnKey, setSortColumnKey] =
		useState<InspectionColumnKey | null>(null);
	const [sortDirection, setSortDirection] = useState<SortDirection | null>(
		null,
	);
	const [currentPage, setCurrentPage] = useState(1);
	const [pageSize, setPageSize] = useState(() => {
		if (typeof window === "undefined") {
			return DEFAULT_INSPECTIONS_PAGE_SIZE;
		}

		return getResponsiveInspectionsPageSize(window.innerHeight);
	});
	const [columnFilters, setColumnFilters] = useState<
		Partial<Record<InspectionColumnKey, string>>
	>({});
	const [areTableViewSettingsHydrated, setAreTableViewSettingsHydrated] =
		useState(() => !tableViewStorageKey);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		if (!tableViewStorageKey) {
			setAreTableViewSettingsHydrated(true);
			return;
		}

		const storage =
			tableViewStorageArea === "localStorage"
				? window.localStorage
				: window.sessionStorage;
		const raw = storage.getItem(tableViewStorageKey);

		if (!raw) {
			setAreTableViewSettingsHydrated(true);
			return;
		}

		try {
			const parsed = JSON.parse(raw) as {
				selectedInspectionView?: unknown;
				hiddenColumnsByView?: unknown;
			};
			const allColumnKeys = new Set(getAllInspectionColumnKeys());
			const normalizeHiddenColumns = (value: unknown) =>
				Array.isArray(value)
					? value.filter(
							(columnKey): columnKey is InspectionColumnKey =>
								typeof columnKey === "string" &&
								allColumnKeys.has(columnKey as InspectionColumnKey),
					  )
					: [];
			const parsedHiddenColumnsByView =
				parsed.hiddenColumnsByView &&
				typeof parsed.hiddenColumnsByView === "object" &&
				!Array.isArray(parsed.hiddenColumnsByView)
					? (parsed.hiddenColumnsByView as Partial<
						Record<InspectionViewId, unknown>
					  >)
					: {};
			const nextHiddenColumnsByView: HiddenColumnsByView = {
				calosc: normalizeHiddenColumns(parsedHiddenColumnsByView.calosc),
				podstawowe: normalizeHiddenColumns(parsedHiddenColumnsByView.podstawowe),
				terminy: normalizeHiddenColumns(parsedHiddenColumnsByView.terminy),
			};

			const nextSelectedView: InspectionViewId =
				parsed.selectedInspectionView === "podstawowe" ||
				parsed.selectedInspectionView === "terminy" ||
				parsed.selectedInspectionView === "calosc"
					? parsed.selectedInspectionView
					: "calosc";

			setHiddenColumnsByView(nextHiddenColumnsByView);
			setSelectedInspectionView(nextSelectedView);
			setHiddenColumns(nextHiddenColumnsByView[nextSelectedView]);
		} catch {
			setHiddenColumnsByView(createDefaultHiddenColumnsByView());
			setSelectedInspectionView("calosc");
			setHiddenColumns(getDefaultHiddenColumnsForView("calosc"));
		}

		setAreTableViewSettingsHydrated(true);
	}, [tableViewStorageArea, tableViewStorageKey]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		if (!tableViewStorageKey || !areTableViewSettingsHydrated) {
			return;
		}

		const storage =
			tableViewStorageArea === "localStorage"
				? window.localStorage
				: window.sessionStorage;
		const nextHiddenColumnsByView: HiddenColumnsByView = {
			...hiddenColumnsByView,
			[selectedInspectionView]: hiddenColumns,
		};

		storage.setItem(
			tableViewStorageKey,
			JSON.stringify({
				selectedInspectionView,
				hiddenColumnsByView: nextHiddenColumnsByView,
			}),
		);
	}, [
		areTableViewSettingsHydrated,
		hiddenColumns,
		hiddenColumnsByView,
		selectedInspectionView,
		tableViewStorageArea,
		tableViewStorageKey,
	]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		const updatePageSize = () => {
			setPageSize(getResponsiveInspectionsPageSize(window.innerHeight));
		};

		updatePageSize();
		window.addEventListener("resize", updatePageSize);

		return () => {
			window.removeEventListener("resize", updatePageSize);
		};
	}, []);

	const visibleInspectionColumns = useMemo(
		() =>
			INSPECTION_COLUMNS.map((column) => column.key).filter(
				(columnKey) => !hiddenColumns.includes(columnKey),
			),
		[hiddenColumns],
	);

	const visibleInspectionColumnDefinitions = useMemo(
		() =>
			INSPECTION_COLUMNS.filter((column) =>
				visibleInspectionColumns.includes(column.key),
			),
		[visibleInspectionColumns],
	);

	const selectableColumnDefinitions = useMemo(
		() => INSPECTION_COLUMNS,
		[],
	);

	const draftVisibleInspectionColumns = useMemo(
		() =>
			INSPECTION_COLUMNS.map((column) => column.key).filter(
				(columnKey) => !draftHiddenColumns.includes(columnKey),
			),
		[draftHiddenColumns],
	);

	const draftSelectableColumnDefinitions = useMemo(
		() => INSPECTION_COLUMNS,
		[],
	);

	const hasActiveFilters = useMemo(
		() => hasActiveTableFilters(columnFilters, advancedFilters),
		[advancedFilters, columnFilters],
	);

	const canClearFilters = hasActiveFilters;

	const advancedFilterValuesByColumn = useMemo(() => {
		const map: Partial<Record<InspectionColumnKey, string[]>> = {};

		INSPECTION_COLUMNS.forEach((column) => {
			const uniqueValues = Array.from(
				new Set(
					inspectionRows.flatMap((row) =>
						splitInspectionAdvancedFilterCellValue(
							column.key,
							toSafeString(row[column.key]),
						),
					),
				),
			).sort((left, right) =>
				left.localeCompare(right, "pl", { sensitivity: "base", numeric: true }),
			);

			map[column.key] = uniqueValues;
		});

		return map;
	}, [inspectionRows]);

	const selectedAdvancedFilterValues = useMemo(
		() =>
			(advancedFilters[advancedFilterColumnKey] ?? []).filter(
				(value) => !isAdvancedDateRangeFilterToken(value),
			),
		[advancedFilterColumnKey, advancedFilters],
	);

	const selectedAdvancedFilterDateRange = useMemo(
		() =>
			getAdvancedDateRangeFromSelectedValues(
				advancedFilters[advancedFilterColumnKey] ?? [],
			),
		[advancedFilterColumnKey, advancedFilters],
	);

	const visibleAdvancedFilterValues = useMemo(() => {
		const sourceValues =
			advancedFilterValuesByColumn[advancedFilterColumnKey] ?? [];
		const normalizedSearch = advancedFilterSearch.trim().toLowerCase();

		if (!normalizedSearch) {
			return sourceValues;
		}

		return sourceValues.filter((value) =>
			value.toLowerCase().includes(normalizedSearch),
		);
	}, [
		advancedFilterColumnKey,
		advancedFilterSearch,
		advancedFilterValuesByColumn,
	]);

	const filteredAndSortedInspectionRows = useMemo(() => {
		const filteredRows = inspectionRows.filter((row) => {
			const advancedFiltersMatch = Object.entries(advancedFilters).every(
				([rawColumnKey, selectedValues]) => {
					const columnKey = rawColumnKey as InspectionColumnKey;
					return matchesInspectionAdvancedFilterCellValue(
						columnKey,
						toSafeString(row[columnKey]),
						Array.isArray(selectedValues) ? selectedValues : [],
					);
				},
			);

			if (!advancedFiltersMatch) {
				return false;
			}

			return Object.entries(columnFilters).every(
				([rawColumnKey, rawFilterValue]) => {
					const filterValue = rawFilterValue?.trim();
					if (!filterValue) {
						return true;
					}

					const columnKey = rawColumnKey as InspectionColumnKey;
					const cellValue = toSafeString(row[columnKey]).toLowerCase();
					return cellValue.includes(filterValue.toLowerCase());
				},
			);
		});

		if (!sortColumnKey || !sortDirection) {
			return filteredRows;
		}

		const directionMultiplier = sortDirection === "asc" ? 1 : -1;
		return [...filteredRows].sort((leftRow, rightRow) => {
			if (sortColumnKey === "lp") {
				return (leftRow.lp - rightRow.lp) * directionMultiplier;
			}

			const leftValue = toSafeString(leftRow[sortColumnKey]);
			const rightValue = toSafeString(rightRow[sortColumnKey]);

			return (
				leftValue.localeCompare(rightValue, "pl", {
					numeric: true,
					sensitivity: "base",
				}) * directionMultiplier
			);
		});
	}, [
		inspectionRows,
		advancedFilters,
		columnFilters,
		sortColumnKey,
		sortDirection,
	]);

	const totalPages = useMemo(
		() =>
			Math.max(
				1,
				Math.ceil(filteredAndSortedInspectionRows.length / pageSize),
			),
		[filteredAndSortedInspectionRows.length, pageSize],
	);

	useEffect(() => {
		setCurrentPage(totalPages);
	}, [
		advancedFilters,
		columnFilters,
		hiddenColumns,
		totalPages,
		sortColumnKey,
		sortDirection,
		selectedInspectionView,
	]);

	useEffect(() => {
		setCurrentPage((previous) => Math.min(Math.max(previous, 1), totalPages));
	}, [totalPages]);

	const paginatedInspectionRows = useMemo(() => {
		const startIndex = (currentPage - 1) * pageSize;
		const endIndex = startIndex + pageSize;
		return filteredAndSortedInspectionRows.slice(startIndex, endIndex);
	}, [currentPage, filteredAndSortedInspectionRows, pageSize]);

	const paginationItems = useMemo(
		() => buildPaginationItems(currentPage, totalPages),
		[currentPage, totalPages],
	);

	const handlePageChange = (nextPage: number) => {
		if (!Number.isFinite(nextPage)) {
			return;
		}

		const normalizedPage = Math.trunc(nextPage);
		const boundedPage = Math.min(Math.max(normalizedPage, 1), totalPages);
		setCurrentPage(boundedPage);
	};

	const handlePageSizeChange = (nextPageSize: number) => {
		if (!Number.isFinite(nextPageSize)) {
			return;
		}

		const normalizedPageSize = Math.max(1, Math.trunc(nextPageSize));
		setPageSize(normalizedPageSize);
	};

	const handleSortByColumn = (columnKey: InspectionColumnKey) => {
		if (sortColumnKey !== columnKey) {
			setSortColumnKey(columnKey);
			setSortDirection("asc");
			return;
		}

		if (sortDirection === "asc") {
			setSortDirection("desc");
			return;
		}

		if (sortDirection === "desc") {
			setSortColumnKey(null);
			setSortDirection(null);
			return;
		}

		setSortDirection("asc");
	};

	const handleFilterChange = (
		columnKey: InspectionColumnKey,
		value: string,
	) => {
		setColumnFilters((prev) => ({ ...prev, [columnKey]: value }));
	};

	const clearFilters = () => {
		setColumnFilters({});
		setAdvancedFilters({});
	};

	const toggleAdvancedFilterValue = (value: string) => {
		setAdvancedFilters((prev) => {
			const currentValues = (prev[advancedFilterColumnKey] ?? []).filter(
				(item) => !isAdvancedDateRangeFilterToken(item),
			);
			const nextValues = currentValues.includes(value)
				? currentValues.filter((item) => item !== value)
				: [...currentValues, value];

			return {
				...prev,
				[advancedFilterColumnKey]: nextValues,
			};
		});
	};

	const selectAllVisibleAdvancedFilterValues = () => {
		setAdvancedFilters((prev) => {
			const currentValues = new Set(
				(prev[advancedFilterColumnKey] ?? []).filter(
					(value) => !isAdvancedDateRangeFilterToken(value),
				),
			);
			visibleAdvancedFilterValues.forEach((value) => currentValues.add(value));

			return {
				...prev,
				[advancedFilterColumnKey]: Array.from(currentValues),
			};
		});
	};

	const clearAdvancedFilterForSelectedColumn = () => {
		setAdvancedFilters((prev) => ({
			...prev,
			[advancedFilterColumnKey]: [],
		}));
	};

	const setAdvancedFilterDateRange = (range: { from: string; to: string }) => {
		setAdvancedFilters((prev) => {
			const token = createAdvancedDateRangeFilterToken(range);

			return {
				...prev,
				[advancedFilterColumnKey]: token ? [token] : [],
			};
		});
	};

	const openAdvancedFilterForColumn = (
		columnKey: InspectionColumnKey,
		triggerElement: HTMLElement,
	) => {
		setAdvancedFilterAnchor(getFloatingPanelAnchor(triggerElement));
		setAdvancedFilterColumnKey(columnKey);
		setAdvancedFilterSearch("");
		setIsAdvancedFilterModalOpen(true);
	};

	const hideColumnInDraft = (columnKey: InspectionColumnKey) => {
		if (draftHiddenColumns.includes(columnKey)) {
			return;
		}

		const nextHiddenColumns = [...draftHiddenColumns, columnKey];
		setDraftHiddenColumns(nextHiddenColumns);
		setDraftHiddenColumnsByView((prev) => ({
			...prev,
			[draftSelectedInspectionView]: nextHiddenColumns,
		}));
	};

	const handleDraftColumnVisibilityChange = (
		columnKey: InspectionColumnKey,
		isVisible: boolean,
	) => {
		if (!isVisible && draftVisibleInspectionColumns.length <= 1) {
			return;
		}

		if (isVisible) {
			const nextHiddenColumns = draftHiddenColumns.filter(
				(key) => key !== columnKey,
			);
			setDraftHiddenColumns(nextHiddenColumns);
			setDraftHiddenColumnsByView((prev) => ({
				...prev,
				[draftSelectedInspectionView]: nextHiddenColumns,
			}));
			return;
		}

		hideColumnInDraft(columnKey);
	};

	const handleOpenViewModal = () => {
		const nextDraftHiddenByView = {
			...hiddenColumnsByView,
			[selectedInspectionView]: hiddenColumns,
		};

		setDraftSelectedInspectionView(selectedInspectionView);
		setDraftHiddenColumns(hiddenColumns);
		setDraftHiddenColumnsByView(nextDraftHiddenByView);
		setIsColumnPickerOpen(true);
	};

	const handleDraftViewSelect = (viewId: InspectionViewId) => {
		const nextDraftHiddenByView = {
			...draftHiddenColumnsByView,
			[draftSelectedInspectionView]: draftHiddenColumns,
		};
		const nextHiddenColumns =
			nextDraftHiddenByView[viewId] ?? getDefaultHiddenColumnsForView(viewId);

		setDraftHiddenColumnsByView(nextDraftHiddenByView);
		setDraftSelectedInspectionView(viewId);
		setDraftHiddenColumns(nextHiddenColumns);
	};

	const handleDraftSelectAllColumns = () => {
		setDraftHiddenColumns([]);
		setDraftHiddenColumnsByView((prev) => ({
			...prev,
			[draftSelectedInspectionView]: [],
		}));
	};

	const handleDraftDeselectAllColumns = () => {
		const nextHiddenColumns = getAllInspectionColumnKeys();
		setDraftHiddenColumns(nextHiddenColumns);
		setDraftHiddenColumnsByView((prev) => ({
			...prev,
			[draftSelectedInspectionView]: nextHiddenColumns,
		}));
	};

	const handleDraftResetSelection = () => {
		const nextHiddenColumns = getDefaultHiddenColumnsForView(
			draftSelectedInspectionView,
		);
		setDraftHiddenColumns(nextHiddenColumns);
		setDraftHiddenColumnsByView((prev) => ({
			...prev,
			[draftSelectedInspectionView]: nextHiddenColumns,
		}));
	};

	const handleApplyViewChanges = () => {
		const nextHiddenByView = {
			...draftHiddenColumnsByView,
			[draftSelectedInspectionView]: draftHiddenColumns,
		};
		const nextHiddenColumns =
			nextHiddenByView[draftSelectedInspectionView] ??
			getDefaultHiddenColumnsForView(draftSelectedInspectionView);

		setHiddenColumnsByView(nextHiddenByView);
		setSelectedInspectionView(draftSelectedInspectionView);
		setHiddenColumns(nextHiddenColumns);
		setIsColumnPickerOpen(false);
	};

	return {
		advancedFilterAnchor,
		advancedFilterColumnKey,
		advancedFilterSearch,
		advancedFilters,
		clearAdvancedFilterForSelectedColumn,
		clearFilters,
		columnFilters,
		draftHiddenColumns,
		draftSelectableColumnDefinitions,
		draftVisibleInspectionColumnsCount: draftVisibleInspectionColumns.length,
		draftSelectedInspectionView,
		filteredAndSortedInspectionRows,
		paginatedInspectionRows,
		currentPage,
		totalPages,
		paginationItems,
		handlePageChange,
		handlePageSizeChange,
		pageSize,
		handleApplyViewChanges,
		handleDraftColumnVisibilityChange,
		handleDraftDeselectAllColumns,
		handleDraftResetSelection,
		handleDraftSelectAllColumns,
		handleDraftViewSelect,
		handleFilterChange,
		handleOpenViewModal,
		handleSortByColumn,
		hasActiveFilters,
		canClearFilters,
		isAdvancedFilterModalOpen,
		isColumnPickerOpen,
		selectedAdvancedFilterDateRange,
		openAdvancedFilterForColumn,
		selectableColumnDefinitions,
		selectedAdvancedFilterValues,
		selectedInspectionView,
		selectAllVisibleAdvancedFilterValues,
		setAdvancedFilterSearch,
		setAdvancedFilterDateRange,
		setDraftHiddenColumns,
		setIsAdvancedFilterModalOpen,
		setIsColumnPickerOpen,
		sortColumnKey,
		sortDirection,
		toggleAdvancedFilterValue,
		visibleAdvancedFilterValues,
		visibleInspectionColumnDefinitions,
	};
}
