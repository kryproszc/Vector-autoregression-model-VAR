"use client";

import { ChevronDown, Download } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchInspectionsReportMatrix } from "@/features/reports/api";
import type { ReportInspectionMatrixRow } from "@/features/reports/types";
import {
	createStyledExportWorkbook,
	saveWorkbookAsXlsx,
} from "@/shared/utils/excel-export";

type ReportsPanelProps = {
	operatorLogin: string;
};

const INSPECTIONS_CHANGED_EVENT = "inspections:changed";

function TableLoadingOverlay() {
	return (
		<div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
			<div className="flex items-center gap-3 rounded-full border border-slate-200 bg-white/90 px-4 py-2 shadow-[0_8px_20px_rgba(2,8,23,0.15)]">
				<span className="h-4 w-4 animate-spin rounded-full border-2 border-[#7aa5dc] border-t-[#255087]" />
				<span className="font-medium text-slate-700 text-sm">Ładowanie danych...</span>
			</div>
		</div>
	);
}

export function ReportsPanel({ operatorLogin }: ReportsPanelProps) {
	const [rows, setRows] = useState<ReportInspectionMatrixRow[]>([]);
	const [years, setYears] = useState<string[]>([]);
	const [isLoading, setIsLoading] = useState(true);
	const [isExporting, setIsExporting] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [isHeatmapEnabled, setIsHeatmapEnabled] = useState(false);
	const [selectedPlants, setSelectedPlants] = useState<string[]>([]);
	const [selectedEntityTypes, setSelectedEntityTypes] = useState<string[]>([]);
	const [selectedYears, setSelectedYears] = useState<string[]>([]);
	const [plantSearch, setPlantSearch] = useState("");
	const [entityTypeSearch, setEntityTypeSearch] = useState("");
	const [yearSearch, setYearSearch] = useState("");
	const filtersRef = useRef<HTMLDivElement | null>(null);
	const codeColumnWidthCh = 14;
	const firstColumnNaturalWidthCh = useMemo(() => {
		const headerLength = "Nazwa podmiotu".length;
		const longestNameLength = rows.reduce(
			(maxLength, row) => Math.max(maxLength, row.nazwaPodmiotu.length),
			headerLength,
		);

		return Math.min(32, Math.max(18, longestNameLength + 2));
	}, [rows]);

	const toControlLabelFromCellEntry = useCallback(
		(entry: ReportInspectionMatrixRow["cells"][string][number]) => {
			const normalizedScopes = entry.scopes
				.map((scope) => scope.trim())
				.filter(Boolean);

			if (normalizedScopes.length === 0) {
				return `${entry.type}_[]`;
			}

			return `${entry.type}_[${normalizedScopes.join(", ")}]`;
		},
		[],
	);

	const parseControls = (value: string) =>
		value
			.split(",")
			.map((item) => item.trim())
			.filter((item) => item.length > 0 && item !== "-");

	const getControlsForCell = useCallback(
		(row: ReportInspectionMatrixRow, year: string) => {
			const cellEntries = row.cells[year] ?? [];
			if (cellEntries.length > 0) {
				return cellEntries.map(toControlLabelFromCellEntry);
			}

			return parseControls(row.wartosci[year] ?? "-");
		},
		[toControlLabelFromCellEntry],
	);

	const selectedPlantsSet = useMemo(() => new Set(selectedPlants), [selectedPlants]);
	const selectedEntityTypesSet = useMemo(
		() => new Set(selectedEntityTypes),
		[selectedEntityTypes],
	);
	const selectedYearsSet = useMemo(() => new Set(selectedYears), [selectedYears]);

	const allPlants = useMemo(
		() =>
			Array.from(new Set(rows.map((row) => row.nazwaPodmiotu))).sort((left, right) =>
				left.localeCompare(right),
			),
		[rows],
	);

	const allEntityTypes = useMemo(
		() =>
			Array.from(
				new Set(
					rows
						.map((row) => row.rodzajPodmiotu.trim())
						.filter((value) => value.length > 0 && value !== "-"),
				),
			).sort((left, right) => left.localeCompare(right, "pl", { sensitivity: "base" })),
		[rows],
	);

	const allYears = useMemo(() => [...years], [years]);

	const visibleYears = useMemo(() => {
		if (selectedYearsSet.size === 0) {
			return allYears;
		}

		return allYears.filter((year) => selectedYearsSet.has(year));
	}, [allYears, selectedYearsSet]);

	const yearColumnWidthByYearPx = useMemo(() => {
		const next: Record<string, number> = {};

		for (const year of visibleYears) {
			let maxLabelLength = year.length;

			for (const row of rows) {
				const controls = getControlsForCell(row, year);
				for (const control of controls) {
					maxLabelLength = Math.max(maxLabelLength, control.length);
				}
			}

			next[year] = Math.max(160, Math.round((maxLabelLength + 6) * 7));
		}

		return next;
	}, [getControlsForCell, rows, visibleYears]);

	const firstColumnWidthCh = useMemo(() => {
		const visibleYearCount = visibleYears.length;
		const maxByYearCount =
			visibleYearCount <= 4
				? 30
				: visibleYearCount <= 6
					? 26
					: visibleYearCount <= 8
						? 23
						: visibleYearCount <= 10
							? 20
							: 18;

		return Math.max(16, Math.min(firstColumnNaturalWidthCh, maxByYearCount));
	}, [firstColumnNaturalWidthCh, visibleYears.length]);

	const visiblePlantOptions = useMemo(() => {
		const query = plantSearch.trim().toLowerCase();
		if (!query) {
			return allPlants;
		}

		return allPlants.filter((plant) => plant.toLowerCase().includes(query));
	}, [allPlants, plantSearch]);

	const visibleEntityTypeOptions = useMemo(() => {
		const query = entityTypeSearch.trim().toLowerCase();
		if (!query) {
			return allEntityTypes;
		}

		return allEntityTypes.filter((entityType) =>
			entityType.toLowerCase().includes(query),
		);
	}, [allEntityTypes, entityTypeSearch]);

	const visibleYearOptions = useMemo(() => {
		const query = yearSearch.trim();
		if (!query) {
			return allYears;
		}

		return allYears.filter((year) => year.includes(query));
	}, [allYears, yearSearch]);

	const getSelectionLabel = useCallback(
		(label: string, selectedCount: number, total: number) => {
			if (total === 0) return `${label}: brak`;
			if (selectedCount === 0) return `${label}: wszystkie`;
			return `${label}: ${selectedCount}`;
		},
		[],
	);

	const filteredRows = useMemo(() => {
		return rows.filter((row) => {
			if (selectedPlantsSet.size > 0 && !selectedPlantsSet.has(row.nazwaPodmiotu)) {
				return false;
			}

			if (
				selectedEntityTypesSet.size > 0 &&
				!selectedEntityTypesSet.has(row.rodzajPodmiotu)
			) {
				return false;
			}

			return true;
		});
	}, [
		rows,
		selectedPlantsSet,
		selectedEntityTypesSet,
	]);

	const togglePlant = useCallback((plant: string) => {
		setSelectedPlants((previous) =>
			previous.includes(plant)
				? previous.filter((item) => item !== plant)
				: [...previous, plant],
		);
	}, []);

	const toggleEntityType = useCallback((entityType: string) => {
		setSelectedEntityTypes((previous) =>
			previous.includes(entityType)
				? previous.filter((item) => item !== entityType)
				: [...previous, entityType],
		);
	}, []);

	const toggleYear = useCallback((year: string) => {
		setSelectedYears((previous) =>
			previous.includes(year)
				? previous.filter((item) => item !== year)
				: [...previous, year],
		);
	}, []);

	const closeOpenFilters = useCallback(() => {
		if (!filtersRef.current) return;

		for (const detailsElement of filtersRef.current.querySelectorAll("details[open]")) {
			(detailsElement as HTMLDetailsElement).open = false;
		}
	}, []);

	const clearFilters = useCallback(() => {
		setSelectedPlants([]);
		setSelectedEntityTypes([]);
		setSelectedYears([]);
		setIsHeatmapEnabled(false);
		closeOpenFilters();
	}, [closeOpenFilters]);

	const handleFilterToggle = useCallback((currentDetails: HTMLDetailsElement) => {
		if (!currentDetails.open || !filtersRef.current) return;

		for (const detailsElement of filtersRef.current.querySelectorAll("details[open]")) {
			if (detailsElement !== currentDetails) {
				(detailsElement as HTMLDetailsElement).open = false;
			}
		}
	}, []);

	useEffect(() => {
		setSelectedPlants((previous) => {
			const filtered = previous.filter((plant) => allPlants.includes(plant));
			return filtered.length === previous.length ? previous : filtered;
		});
	}, [allPlants]);

	useEffect(() => {
		setSelectedEntityTypes((previous) => {
			const filtered = previous.filter((entityType) =>
				allEntityTypes.includes(entityType),
			);
			return filtered.length === previous.length ? previous : filtered;
		});
	}, [allEntityTypes]);

	useEffect(() => {
		setSelectedYears((previous) => {
			const filtered = previous.filter((year) => allYears.includes(year));
			return filtered.length === previous.length ? previous : filtered;
		});
	}, [allYears]);

	const getCellIntensityClass = (controlsCount: number) => {
		if (!isHeatmapEnabled) return "";
		if (controlsCount <= 0) return "";
		if (controlsCount === 1) return "bg-[#e9f2ff]";
		if (controlsCount === 2) return "bg-[#d4e7ff]";
		return "bg-[#bdd9ff]";
	};

	const getDisplayCellValue = useCallback(
		(row: ReportInspectionMatrixRow, year: string) => {
			const controls = getControlsForCell(row, year);
			if (controls.length === 0) {
				return "-";
			}

			return controls.join(", ");
		},
		[getControlsForCell],
	);

	const parseControlLabel = useCallback((controlLabel: string) => {
		const normalized = controlLabel.trim();
		const match = normalized.match(/^([A-Za-z]+)_\[(.*)\]$/);
		if (!match) {
			return {
				type: normalized.toUpperCase().startsWith("WN") ? "WN" : "K",
				scopesLabel: normalized || "-",
			};
		}

		const rawType = (match[1] ?? "K").trim().toUpperCase();
		const type = rawType === "WN" ? "WN" : "K";
		const rawScopes = (match[2] ?? "").trim();
		const scopes = rawScopes
			.split(",")
			.map((scope) => scope.trim())
			.filter(Boolean);

		return {
			type,
			scopesLabel: scopes.length > 0 ? scopes.join(", ") : "Brak zakresu",
		};
	}, []);

	const getControlTypeRank = useCallback((controlLabel: string) => {
		const parsed = parseControlLabel(controlLabel);
		return parsed.type === "K" ? 0 : 1;
	}, [parseControlLabel]);

	const getCellTooltip = useCallback(
		(row: ReportInspectionMatrixRow, year: string) => {
			const controls = getControlsForCell(row, year);
			if (controls.length === 0) {
				return "-";
			}

			return controls.join("\n");
		},
		[getControlsForCell],
	);

	const loadReport = useCallback(async () => {
		setIsLoading(true);
		setError(null);

		const result = await fetchInspectionsReportMatrix(operatorLogin);
		if (!result.ok) {
			setRows([]);
			setYears([]);
			setError(result.error);
			setIsLoading(false);
			return;
		}

		setRows(result.data.rows);
		setYears(result.data.lata);
		setIsLoading(false);
	}, [operatorLogin]);

	const handleExportToExcel = useCallback(async () => {
		if (isExporting || filteredRows.length === 0) {
			return;
		}

		setIsExporting(true);
		setError(null);

		try {
			const workbook = await createStyledExportWorkbook("Mapa inspekcji");
			const worksheet = workbook.addWorksheet("Mapa inspekcji");

			const headers = ["Kod inspekcji", "Nazwa podmiotu", ...visibleYears];
			worksheet.addRow(headers);

			for (const row of filteredRows) {
				const values = visibleYears.map((year) => {
					return getDisplayCellValue(row, year);
				});
				worksheet.addRow([row.kodInspekcji || "-", row.nazwaPodmiotu, ...values]);
			}

			const headerRow = worksheet.getRow(1);
			headerRow.font = { bold: true };
			headerRow.alignment = { vertical: "middle", horizontal: "left" };

			worksheet.columns = headers.map((header, index) => {
				if (index === 0) {
					return { header, width: Math.max(14, codeColumnWidthCh) };
				}
				if (index === 1) {
					return { header, width: Math.max(22, firstColumnWidthCh) };
				}
				return {
					header,
					width: Math.max(
						12,
						Math.round((yearColumnWidthByYearPx[header] ?? 160) / 8),
					),
				};
			});

			const fileName = "wykonane-inspekcje.xlsx";
			await saveWorkbookAsXlsx(workbook, fileName);
		} catch (exportError) {
			if (exportError instanceof DOMException && exportError.name === "AbortError") {
				return;
			}

			setError("Nie udało się wyeksportować danych do Excela.");
		} finally {
			setIsExporting(false);
		}
	}, [
		codeColumnWidthCh,
		filteredRows,
		firstColumnWidthCh,
		getDisplayCellValue,
		isExporting,
		yearColumnWidthByYearPx,
		visibleYears,
	]);

	useEffect(() => {
		loadReport();
	}, [loadReport]);

	useEffect(() => {
		const handleInspectionsChanged = () => {
			void loadReport();
		};

		window.addEventListener(INSPECTIONS_CHANGED_EVENT, handleInspectionsChanged);
		return () => {
			window.removeEventListener(INSPECTIONS_CHANGED_EVENT, handleInspectionsChanged);
		};
	}, [loadReport]);

	useEffect(() => {
		const handlePointerDown = (event: MouseEvent) => {
			if (!filtersRef.current) return;

			const target = event.target;
			if (!(target instanceof Node)) return;

			if (!filtersRef.current.contains(target)) {
				closeOpenFilters();
			}
		};

		const handleEscape = (event: KeyboardEvent) => {
			if (event.key === "Escape") {
				closeOpenFilters();
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		document.addEventListener("keydown", handleEscape);

		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
			document.removeEventListener("keydown", handleEscape);
		};
	}, [closeOpenFilters]);

	return (
		<section className="flex h-[calc(100vh-4.4rem)] min-h-0 flex-col space-y-4 rounded-2xl border border-slate-700/70 bg-[#101f39] p-5 shadow-[0_18px_44px_rgba(2,8,23,0.34)]">
			<div className="border-slate-700/70 border-b pb-2">
				<div className="flex flex-wrap items-center justify-between gap-3">
					<div>
						<h2 className="font-semibold text-xl text-slate-100">Mapa inspekcji</h2>
					</div>

					<div className="flex items-center gap-2">
						<button
							type="button"
							onClick={handleExportToExcel}
							disabled={isLoading || isExporting || filteredRows.length === 0}
							className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#6ea3f0] bg-[#285186] px-3.5 font-medium text-sm text-slate-100 transition-colors hover:bg-[#3563a1] disabled:cursor-not-allowed disabled:opacity-70"
						>
							<Download size={14} />
							{isExporting ? "Eksportowanie..." : "Eksportuj"}
						</button>
					</div>
				</div>
			</div>

			<div ref={filtersRef} className="space-y-2 rounded-lg border border-slate-700/70 bg-[#122c4e] p-3">
				<div className="flex flex-wrap items-center justify-between gap-2">
					<div className="flex flex-wrap items-center gap-2">
						<details
							className="group relative"
							onToggle={(event) => handleFilterToggle(event.currentTarget)}
						>
							<summary className="inline-flex w-52 cursor-pointer list-none items-center justify-between rounded-md border border-slate-300 bg-white px-3 py-2 font-medium text-slate-800 text-sm transition-colors hover:bg-slate-50">
								<span className="truncate">
									{getSelectionLabel(
										"Rodzaj podmiotu",
										selectedEntityTypes.length,
										allEntityTypes.length,
									)}
								</span>
								<ChevronDown
									size={14}
									className="shrink-0 text-slate-500 transition-transform duration-150 group-open:rotate-180"
								/>
							</summary>
							<div className="absolute left-0 z-30 mt-2 w-80 rounded-lg border border-slate-300 bg-slate-100 p-2 shadow-[0_14px_28px_rgba(2,8,23,0.24)]">
								<input
									type="text"
									value={entityTypeSearch}
									onChange={(event) => setEntityTypeSearch(event.target.value)}
									placeholder="Szukaj rodzaju podmiotu"
									className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-2 text-slate-700 text-sm outline-none placeholder:text-slate-400 focus:border-slate-400"
								/>
								<div className="subtle-vertical-scroll mt-2 max-h-36 space-y-1 overflow-auto pr-1">
									{visibleEntityTypeOptions.map((entityType) => {
										const isSelected = selectedEntityTypes.includes(entityType);
										return (
											<label
												key={entityType}
												className="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-slate-700 text-sm hover:bg-slate-200/70"
												title={entityType}
											>
												<input
													type="checkbox"
													checked={isSelected}
													onChange={() => toggleEntityType(entityType)}
													className="h-3.5 w-3.5"
												/>
												<span className="truncate">{entityType}</span>
											</label>
										);
									})}
									{visibleEntityTypeOptions.length === 0 ? (
										<p className="px-2 py-1 text-slate-500 text-sm">Brak wyników.</p>
									) : null}
								</div>
							</div>
						</details>

						<details
							className="group relative"
							onToggle={(event) => handleFilterToggle(event.currentTarget)}
						>
							<summary className="inline-flex w-44 cursor-pointer list-none items-center justify-between rounded-md border border-slate-300 bg-white px-3 py-2 font-medium text-slate-800 text-sm transition-colors hover:bg-slate-50">
								<span className="truncate">
									{getSelectionLabel("Podmioty", selectedPlants.length, allPlants.length)}
								</span>
								<ChevronDown
									size={14}
									className="shrink-0 text-slate-500 transition-transform duration-150 group-open:rotate-180"
								/>
							</summary>
							<div className="absolute left-0 z-30 mt-2 w-80 rounded-lg border border-slate-300 bg-slate-100 p-2 shadow-[0_14px_28px_rgba(2,8,23,0.24)]">
								<input
									type="text"
									value={plantSearch}
									onChange={(event) => setPlantSearch(event.target.value)}
									placeholder="Szukaj zakładu"
									className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-2 text-slate-700 text-sm outline-none placeholder:text-slate-400 focus:border-slate-400"
								/>
								<div className="subtle-vertical-scroll mt-2 max-h-36 space-y-1 overflow-auto pr-1">
									{visiblePlantOptions.map((plant) => {
										const isSelected = selectedPlants.includes(plant);
										return (
											<label
												key={plant}
												className="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-slate-700 text-sm hover:bg-slate-200/70"
												title={plant}
											>
												<input
													type="checkbox"
													checked={isSelected}
													onChange={() => togglePlant(plant)}
													className="h-3.5 w-3.5"
												/>
												<span className="truncate">{plant}</span>
											</label>
										);
									})}
									{visiblePlantOptions.length === 0 ? (
										<p className="px-2 py-1 text-slate-500 text-sm">Brak wyników.</p>
									) : null}
								</div>
							</div>
						</details>

						<details
							className="group relative"
							onToggle={(event) => handleFilterToggle(event.currentTarget)}
						>
							<summary className="inline-flex w-44 cursor-pointer list-none items-center justify-between rounded-md border border-slate-300 bg-white px-3 py-2 font-medium text-slate-800 text-sm transition-colors hover:bg-slate-50">
								<span className="truncate">
									{getSelectionLabel("Rok", selectedYears.length, allYears.length)}
								</span>
								<ChevronDown
									size={14}
									className="shrink-0 text-slate-500 transition-transform duration-150 group-open:rotate-180"
								/>
							</summary>
							<div className="absolute left-0 z-30 mt-2 w-56 rounded-lg border border-slate-300 bg-slate-100 p-2 shadow-[0_14px_28px_rgba(2,8,23,0.24)]">
								<input
									type="text"
									value={yearSearch}
									onChange={(event) => setYearSearch(event.target.value)}
									placeholder="Szukaj roku"
									className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-2 text-slate-700 text-sm outline-none placeholder:text-slate-400 focus:border-slate-400"
								/>
								<div className="subtle-vertical-scroll mt-2 max-h-36 space-y-1 overflow-auto pr-1">
									{visibleYearOptions.map((year) => {
										const isSelected = selectedYears.includes(year);
										return (
											<label
												key={year}
												className="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-slate-700 text-sm hover:bg-slate-200/70"
											>
												<input
													type="checkbox"
													checked={isSelected}
													onChange={() => toggleYear(year)}
													className="h-3.5 w-3.5"
												/>
												<span>{year}</span>
											</label>
										);
									})}
									{visibleYearOptions.length === 0 ? (
										<p className="px-2 py-1 text-slate-500 text-sm">Brak wyników.</p>
									) : null}
								</div>
							</div>
						</details>

						<button
							type="button"
							onClick={() => setIsHeatmapEnabled((prev) => !prev)}
							className={`inline-flex w-44 items-center justify-center rounded-md border px-3 py-2 font-medium text-sm transition-colors ${
								isHeatmapEnabled
									? "border-[#9fc6ff] bg-[#4477b7] text-slate-100"
									: "border-slate-300 bg-white text-slate-800 hover:bg-slate-50"
							}`}
						>
							Intensywność
						</button>

						{isHeatmapEnabled ? (
							<div className="flex flex-wrap items-center gap-1.5 text-xs text-slate-200">
								<span className="rounded-md border border-[#4f6284] bg-[#1a3458] px-2 py-0.5 font-medium">
									Mapa: liczba kontroli w komórce
								</span>
								<span className="rounded-md border border-[#d1dae7] bg-[#f2f5f9] px-1.5 py-0.5 text-[#526278]">
									0
								</span>
								<span className="rounded-md border border-[#c3d8f6] bg-[#e2eeff] px-1.5 py-0.5 text-[#204874]">
									1
								</span>
								<span className="rounded-md border border-[#a6c9f8] bg-[#bbd8ff] px-1.5 py-0.5 text-[#113f75]">
									2
								</span>
								<span className="rounded-md border border-[#79a9e8] bg-[#8dbaf5] px-1.5 py-0.5 text-[#0d2e57]">
									3+
								</span>
							</div>
						) : null}
					</div>

					<div className="flex items-center">
						<button
							type="button"
							onClick={clearFilters}
							disabled={
								selectedPlants.length === 0 &&
								selectedEntityTypes.length === 0 &&
								selectedYears.length === 0 &&
								!isHeatmapEnabled
							}
							className={`inline-flex h-7 items-center rounded px-1.5 transition-colors disabled:cursor-not-allowed ${
								selectedPlants.length > 0 ||
								selectedEntityTypes.length > 0 ||
								selectedYears.length > 0 ||
								isHeatmapEnabled
									? "font-semibold text-blue-300 text-sm hover:text-blue-200"
									: "font-medium text-slate-500 text-xs"
							}`}
						>
							Wyczyść filtry
						</button>
					</div>
				</div>
			</div>

			{error ? (
				<p className="rounded-lg border border-rose-300/60 bg-rose-100/90 px-3 py-2 text-rose-800 text-sm">
					{error}
				</p>
			) : null}

			<div className="relative subtle-horizontal-scroll subtle-vertical-scroll table-scroll-gutter-right min-h-0 flex-1 w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-400 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
				{isLoading ? <TableLoadingOverlay /> : null}
				<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
					<thead>
						<tr className="bg-slate-100 text-slate-800">
							<th
								className="sticky top-0 left-0 z-20 whitespace-nowrap border-slate-300 border-b border-r bg-slate-100 px-3 py-2 text-left font-semibold"
								style={{ width: `${firstColumnWidthCh}ch`, minWidth: `${firstColumnWidthCh}ch` }}
							>
								Nazwa podmiotu
							</th>
							{visibleYears.map((year) => (
								<th
									key={year}
									className="sticky top-0 z-10 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-2.5 py-2 text-left font-semibold"
									style={{
										width: `${yearColumnWidthByYearPx[year] ?? 160}px`,
										minWidth: `${yearColumnWidthByYearPx[year] ?? 160}px`,
										maxWidth: `${yearColumnWidthByYearPx[year] ?? 160}px`,
									}}
								>
									{year}
								</th>
							))}
						</tr>
					</thead>

					<tbody>
						{filteredRows.map((row) => (
							<tr
								key={row.nazwaPodmiotu}
								className="border-slate-200 border-b bg-white transition-colors hover:bg-slate-50 last:border-b-0"
							>
								<td
									className="sticky left-0 z-10 align-top whitespace-nowrap border-slate-200 border-r bg-white px-3 py-2.5 text-slate-900"
									title={row.nazwaPodmiotu}
									style={{ width: `${firstColumnWidthCh}ch`, maxWidth: `${firstColumnWidthCh}ch` }}
								>
									<span className="block truncate">{row.nazwaPodmiotu}</span>
								</td>
								{visibleYears.map((year) => (
									<td
										key={`${row.nazwaPodmiotu}-${year}`}
										className={`px-2.5 py-2.5 align-top whitespace-normal wrap-break-word ${getCellIntensityClass(getControlsForCell(row, year).length)}`}
										title={getCellTooltip(row, year)}
										style={{
											width: `${yearColumnWidthByYearPx[year] ?? 160}px`,
											minWidth: `${yearColumnWidthByYearPx[year] ?? 160}px`,
											maxWidth: `${yearColumnWidthByYearPx[year] ?? 160}px`,
										}}
									>
										{(() => {
											const controls = getControlsForCell(row, year);
											if (controls.length === 0) {
												return "-";
											}

											const sortedControls = [...controls].sort((left, right) => {
												const rankDiff = getControlTypeRank(left) - getControlTypeRank(right);
												if (rankDiff !== 0) {
													return rankDiff;
												}

												return left.localeCompare(right, "pl", { sensitivity: "base" });
											});

											return (
												<div className="space-y-1.5">
													{sortedControls.map((controlLabel, index) => {
														const parsed = parseControlLabel(controlLabel);
														const entryClass = isHeatmapEnabled
															? "rounded-md border border-slate-300/70 bg-transparent px-2 py-1"
															: "rounded-md border border-slate-200 bg-white/85 px-2 py-1";
														const typeBadgeClass =
															parsed.type === "WN"
																? "inline-flex min-w-7 items-center justify-center rounded-md border border-violet-400/80 bg-violet-100 px-1.5 py-0.5 font-bold text-[10px] leading-none tracking-wide text-violet-800"
																: "inline-flex min-w-7 items-center justify-center rounded-md border border-purple-400/80 bg-purple-100 px-1.5 py-0.5 font-bold text-[10px] leading-none tracking-wide text-purple-800";

														return (
															<div
																key={`${controlLabel}-${index}`}
																className={entryClass}
															>
																<div className="flex items-start gap-2">
																<span
																	className={typeBadgeClass}
																>
																	{parsed.type}
																</span>
																<span
																		className="min-w-0 flex-1 break-words text-[12px] leading-[1.3] text-slate-800"
																	title={parsed.scopesLabel}
																>
																	{parsed.scopesLabel}
																</span>
																</div>
															</div>
														);
													})}
												</div>
											);
										})()}
									</td>
								))}
							</tr>
						))}

						{!isLoading && filteredRows.length === 0 ? (
							<tr>
								<td
									colSpan={Math.max(1, visibleYears.length + 1)}
									className="px-3 py-6 text-center text-slate-500 text-sm"
								>
										{selectedPlants.length > 0 ||
									selectedEntityTypes.length > 0 ||
									selectedYears.length > 0
										? "Brak rekordów dla wybranych filtrów."
										: "Brak rekordów raportu."}
								</td>
							</tr>
						) : null}

						{isLoading ? (
							<tr>
								<td
									colSpan={Math.max(1, visibleYears.length + 1)}
									className="px-3 py-6 text-center text-slate-500 text-sm"
								>
									Ładowanie danych raportu...
								</td>
							</tr>
						) : null}
					</tbody>
				</table>
			</div>
		</section>
	);
}
