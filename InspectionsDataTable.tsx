import { type CSSProperties, type ReactNode, useEffect, useRef } from "react";

import { INSPECTION_COLUMN_TOOLTIPS } from "@/features/inspections/data";
import type {
	InspectionColumn,
	InspectionColumnKey,
	InspectionRow,
} from "@/features/inspections/types";
import { RegistryDataTable } from "@/shared/components/table/RegistryDataTable";
import { formatDatesInDisplayText } from "@/shared/utils/date";

type SortDirection = "asc" | "desc";

type InspectionNoLetterFlags = {
	brakDataDoreczeniaPisma: boolean;
	brakDataPismaZastrzezenia: boolean;
	brakDataWyslaniaPismaZZastrzezeniami: boolean;
	brakDataWplywuPisma: boolean;
	brakDataPismaZOdpowiedzia: boolean;
	brakDataWyslaniaPismaZOdpowiedzia: boolean;
};

type InspectionNoAcceptanceDatesFlags = {
	brakDatAkceptacjiNoty: boolean;
};

type InspectionsDataTableProps = {
	rowsError: string | null;
	isRowsLoading: boolean;
	visibleColumns: InspectionColumn[];
	columnWidths?: Partial<Record<InspectionColumnKey, number>>;
	minColumnWidth?: number;
	sortColumnKey: InspectionColumnKey | null;
	sortDirection: SortDirection | null;
	advancedFilters: Partial<Record<InspectionColumnKey, string[]>>;
	columnFilters: Partial<Record<InspectionColumnKey, string>>;
	rows: InspectionRow[];
	noLetterFlagsByRowId?: Record<string, InspectionNoLetterFlags>;
	noAcceptanceDatesByRowId?: Record<string, InspectionNoAcceptanceDatesFlags>;
	statusCodeByRowId?: Record<string, string>;
	selectedInspectionId: string | null;
	flashInspectionId?: string | null;
	centerOnInspectionId?: string | null;
	onInspectionCentered?: () => void;
	onSelectInspection: (inspectionId: string) => void;
	onOpenInspectionPreview?: (inspectionId: string) => void;
	onSortByColumn: (columnKey: InspectionColumnKey) => void;
	onResizeColumn?: (columnKey: InspectionColumnKey, width: number) => void;
	onOpenAdvancedFilter: (
		columnKey: InspectionColumnKey,
		triggerElement: HTMLElement,
	) => void;
	onFilterChange: (columnKey: InspectionColumnKey, value: string) => void;
	footer?: ReactNode;
};

function resolveInspectionTypeFlags(inspectionType: string) {
	const normalizedType = inspectionType.trim().toLowerCase();
	const isControlType =
		normalizedType.includes("kontrol") ||
		normalizedType.startsWith("kont") ||
		normalizedType === "k";
	const isSupervisoryVisitType =
		normalizedType.includes("wizyta") ||
		normalizedType.startsWith("wiz") ||
		normalizedType === "w";

	return {
		isControlType,
		isSupervisoryVisitType,
	};
}

function isNotApplicableByInspectionType(
	inspectionType: string,
	columnKey: InspectionColumnKey,
) {
	const { isControlType, isSupervisoryVisitType } =
		resolveInspectionTypeFlags(inspectionType);

	if (isControlType && !isSupervisoryVisitType) {
		return columnKey === "dataAkceptacjiSprawozdania" || columnKey === "dataDoreczeniaPisma";
	}

	if (isSupervisoryVisitType && !isControlType) {
		return (
			columnKey === "dataDoreczeniaProtokolu" ||
			columnKey === "dataWyslaniaPismaZOdpowiedzia" ||
			columnKey === "dataPismaZOdpowiedzia"
		);
	}

	return false;
}

function resolveStatusRowClass(statusCode: string | undefined) {
	void statusCode;
	return "hover:!bg-slate-50";
}

function resolveStatusRowStyle(statusCode: string | undefined): CSSProperties {
	const normalized = (statusCode ?? "").trim().toUpperCase();
	const statusShade = 100;
	const statusOpacity = 30;
	const supervisoryVisitShade = 100;

	const yellowStatusCodes = new Set([
		"I_SI_5",
		"I_SI_6",
		"I_SI_7",
		"I_SI_8",
		"I_SI_9",
	]);

	const clampOpacity = Math.max(0, Math.min(statusOpacity, 100)) / 100;

	const shadeToRgb = {
		emerald: {
			50: "236 253 245",
			100: "209 250 229",
		},
		sky: {
			50: "240 249 255",
			100: "224 242 254",
		},
		pink: {
			50: "253 242 248",
			100: "252 231 243",
		},
		amber: {
			50: "255 251 235",
			100: "254 243 199",
		},
	} as const;

	if (normalized === "I_SI_2") {
		const rgb = shadeToRgb.emerald[statusShade as 50 | 100] ?? shadeToRgb.emerald[50];
		return {
			backgroundColor: `rgb(${rgb} / ${clampOpacity})`,
		};
	}

	if (normalized === "I_SI_3") {
		const rgb =
			shadeToRgb.sky[supervisoryVisitShade as 50 | 100] ?? shadeToRgb.sky[100];
		return {
			backgroundColor: `rgb(${rgb} / ${clampOpacity})`,
		};
	}

	if (normalized === "I_SI_4") {
		const rgb = shadeToRgb.pink[statusShade as 50 | 100] ?? shadeToRgb.pink[50];
		return {
			backgroundColor: `rgb(${rgb} / ${clampOpacity})`,
		};
	}

	if (yellowStatusCodes.has(normalized)) {
		const rgb = shadeToRgb.amber[statusShade as 50 | 100] ?? shadeToRgb.amber[50];
		return {
			backgroundColor: `rgb(${rgb} / ${clampOpacity})`,
		};
	}

	return {
		backgroundColor: "rgb(255 255 255 / 1)",
	};
}


export function InspectionsDataTable({
	rowsError,
	isRowsLoading,
	visibleColumns,
	columnWidths,
	minColumnWidth,
	sortColumnKey,
	sortDirection,
	advancedFilters,
	columnFilters,
	rows,
	noLetterFlagsByRowId = {},
	noAcceptanceDatesByRowId = {},
	statusCodeByRowId = {},
	selectedInspectionId,
	flashInspectionId = null,
	centerOnInspectionId = null,
	onInspectionCentered,
	onSelectInspection,
	onOpenInspectionPreview,
	onSortByColumn,
	onResizeColumn,
	onOpenAdvancedFilter,
	onFilterChange,
	footer,
}: InspectionsDataTableProps) {
	const tbodyRef = useRef<HTMLTableSectionElement | null>(null);

	useEffect(() => {
		if (!centerOnInspectionId) {
			return;
		}

		const rows = tbodyRef.current?.querySelectorAll<HTMLTableRowElement>(
			"tr[data-inspection-id]",
		);
		if (!rows || rows.length === 0) {
			return;
		}

		const targetRow = Array.from(rows).find(
			(row) => row.dataset.inspectionId === centerOnInspectionId,
		);
		if (!targetRow) {
			return;
		}

		targetRow.scrollIntoView({ block: "center", inline: "nearest" });
		onInspectionCentered?.();
	}, [centerOnInspectionId, onInspectionCentered, rows]);

	return (
		<RegistryDataTable
			isLoading={isRowsLoading}
			errorMessage={rowsError}
			footer={footer}
			containerClassName="-mt-1"
			scrollAreaClassName="h-[calc(96vh-11rem)] min-h-88"
			tableClassName="min-w-350 table-fixed border-collapse text-slate-900 text-sm"
			visibleColumns={visibleColumns.map((column) => ({
				...column,
				tooltip: INSPECTION_COLUMN_TOOLTIPS[column.key],
			}))}
			sortColumnKey={sortColumnKey}
			sortDirection={sortDirection}
			advancedFilters={advancedFilters}
			columnFilters={columnFilters}
			onSortByColumn={onSortByColumn}
			columnWidths={columnWidths}
			minColumnWidth={minColumnWidth}
			wrapHeaderLabels
			controlsInFilterRow
			showInfoIcon
			infoIconSize={11}
			onResizeColumn={onResizeColumn}
			onOpenAdvancedFilter={onOpenAdvancedFilter}
			onFilterChange={onFilterChange}
		>
				<tbody ref={tbodyRef}>
					{rows.map((row) => {
						const isActive = selectedInspectionId === row.id;
						const isFlashing = flashInspectionId === row.id;
						const statusRowClass = resolveStatusRowClass(statusCodeByRowId[row.id]);
						const statusRowStyle = resolveStatusRowStyle(statusCodeByRowId[row.id]);

						return (
							<tr
								key={row.id}
								data-inspection-id={row.id}
								onClick={() => onSelectInspection(row.id)}
								onDoubleClick={() => onOpenInspectionPreview?.(row.id)}
								className={`cursor-pointer border-slate-200 border-b transition-colors last:border-b-0 ${
									isFlashing || isActive
										? "bg-blue-100 text-slate-900 [&>td]:border-blue-300 [&>td]:border-y [&>td:first-child]:border-l [&>td:last-child]:border-r"
										: `${statusRowClass} text-slate-900`
								}`}
								style={isFlashing || isActive ? undefined : statusRowStyle}
							>
								{visibleColumns.map((column) => {
									const value = row[column.key];
									const rawValue = String(value ?? "").trim();
									const normalizedValue = rawValue.toLowerCase();
									const rowFlags = noLetterFlagsByRowId[row.id];
									const acceptanceFlags = noAcceptanceDatesByRowId[row.id];
									const shouldShowNoLetter =
										(column.key === "dataDoreczeniaPisma" &&
											rowFlags?.brakDataDoreczeniaPisma) ||
										(column.key === "dataPismaZastrzezenia" &&
											rowFlags?.brakDataPismaZastrzezenia) ||
										(column.key === "dataWyslaniaPismaZZastrzezeniami" &&
											rowFlags?.brakDataWyslaniaPismaZZastrzezeniami) ||
										(column.key === "dataWplywuPisma" &&
											rowFlags?.brakDataWplywuPisma) ||
										(column.key === "dataPismaZOdpowiedzia" &&
											rowFlags?.brakDataPismaZOdpowiedzia) ||
										(column.key === "dataWyslaniaPismaZOdpowiedzia" &&
											rowFlags?.brakDataWyslaniaPismaZOdpowiedzia);
									const shouldShowNoAcceptanceDates =
										column.key === "dataAkceptacjiNoty" &&
										acceptanceFlags?.brakDatAkceptacjiNoty;
									const shouldShowNotApplicable = isNotApplicableByInspectionType(
										String(row.typInspekcji ?? ""),
										column.key,
									);
									const displayValue =
										shouldShowNotApplicable
											? "Nie dotyczy"
											: shouldShowNoLetter
											? "Brak pisma"
											: shouldShowNoAcceptanceDates
												? "Brak pisma"
												: normalizedValue === "" || normalizedValue === "brak"
												? "-"
												: rawValue;
									const formattedDisplayValue = formatDatesInDisplayText(displayValue);
									const isLongTextColumn =
										column.key === "zakresInspekcji" ||
										column.key === "skladZespolu" ||
										column.key === "szczegolyDotyczaceZakresu" ||
										column.key === "dataAkceptacjiNoty" ||
										column.key === "dataZalecen";
									const isEntityNameColumn = column.key === "nazwaPodmiotu";
									const isScopeDetailsColumn =
										column.key === "szczegolyDotyczaceZakresu";
									const isStatusColumn = column.key === "status";
									const isCommentColumn = column.key === "komentarz";
									const columnWidth = columnWidths?.[column.key];
									const isScopeColumn = column.key === "zakresInspekcji";
									const isTeamColumn = column.key === "skladZespolu";
									const isAcceptanceDatesColumn =
										column.key === "dataAkceptacjiNoty";
									const isRecommendationDatesColumn =
										column.key === "dataZalecen";
									const verticalListItems = isTeamColumn
										? formattedDisplayValue
											.split(";")
											.map((item) => item.trim())
											.filter(Boolean)
										: isScopeColumn
											? formattedDisplayValue
												.split(";")
												.map((item) => item.trim())
												.filter(Boolean)
										: isAcceptanceDatesColumn || isRecommendationDatesColumn
												? formattedDisplayValue
												.split(",")
												.map((item) => item.trim())
												.filter(Boolean)
										: [];
									const shouldUseNumberedList =
										(isScopeColumn && verticalListItems.length > 0) ||
										(isTeamColumn && verticalListItems.length > 1);
									const tooltipListValue = shouldUseNumberedList
										? verticalListItems
												.map((item, index) => `${index + 1}. ${item}`)
												.join("\n")
										: verticalListItems.join("\n");
									const tooltipValue = isLongTextColumn
										? (verticalListItems.length > 0
											? tooltipListValue
											: formattedDisplayValue)
										: undefined;

									return (
										<td
											key={column.key}
											className={`overflow-hidden align-top px-3 py-2.5 whitespace-normal break-words ${
												(column.key === "lp" || column.key === "kodInspekcji") && isActive
													? "font-medium"
													: "font-normal"
											}`}
											title={tooltipValue}
											style={
												columnWidth
													? { width: columnWidth, minWidth: columnWidth, maxWidth: columnWidth }
													: undefined
											}
										>
											{(isTeamColumn || isScopeColumn) && formattedDisplayValue !== "-" ? (
												<div className="subtle-vertical-scroll max-h-28 w-full space-y-1 overflow-y-auto pr-1">
													{shouldUseNumberedList ? (
														<ol className="list-inside list-decimal space-y-1 pl-1">
															{verticalListItems.map((item, index) => (
																<li
																	key={`${row.id}-${column.key}-${index}`}
																	className="whitespace-normal break-words"
																>
																	{item}
																</li>
															))}
														</ol>
													) : (
														verticalListItems.map((item, index) => (
															<div
																key={`${row.id}-${column.key}-${index}`}
																className="whitespace-normal break-words"
															>
																{item}
															</div>
														))
													)}
												</div>
											) : (isAcceptanceDatesColumn || isRecommendationDatesColumn) && formattedDisplayValue !== "-" ? (
												<div className="subtle-vertical-scroll max-h-28 w-full space-y-1 overflow-y-auto pr-1">
													{verticalListItems.map((item, index) => (
														<div key={`${row.id}-${column.key}-${index}`} className="whitespace-normal break-words">
															{item}
														</div>
													))}
												</div>
											) : isEntityNameColumn || isScopeDetailsColumn || isStatusColumn || isCommentColumn ? (
												<div className="w-full whitespace-normal break-words leading-5">
													{shouldShowNotApplicable ? (
														<span className="italic text-slate-400">{formattedDisplayValue}</span>
													) : (
														formattedDisplayValue
													)}
												</div>
											) : isLongTextColumn ? (
												<div className="w-full whitespace-normal break-words leading-5">
													{shouldShowNotApplicable ? (
														<span className="italic text-slate-400">{formattedDisplayValue}</span>
													) : (
														formattedDisplayValue
													)}
												</div>
											) : (
												<div className="w-full whitespace-normal break-words leading-5">
													{shouldShowNotApplicable ? (
														<span className="italic text-slate-400">{formattedDisplayValue}</span>
													) : (
														formattedDisplayValue
													)}
												</div>
											)}
										</td>
									);
								})}
							</tr>
						);
					})}

					{rows.length === 0 ? (
						<tr>
							<td
								colSpan={visibleColumns.length}
								className="px-3 py-6 text-center text-slate-500 text-sm"
							>
								Brak rekordów spełniających aktualne filtry.
							</td>
						</tr>
					) : null}
				</tbody>
		</RegistryDataTable>
	);
}
