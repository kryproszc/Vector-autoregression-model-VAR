import { useEffect } from "react";
import type { ReactNode } from "react";

import { INSPECTION_COLUMN_TOOLTIPS } from "@/features/inspections/data";
import type {
	InspectionColumn,
	InspectionColumnKey,
	InspectionRow,
} from "@/features/inspections/types";
import { RegistryDataTable } from "@/shared/components/table/RegistryDataTable";

type SortDirection = "asc" | "desc";

type InspectionNoLetterFlags = {
	brakDataDoreczeniaPisma: boolean;
	brakDataPismaZastrzezenia: boolean;
	brakDataWplywuPisma: boolean;
	brakDataPismaZOdpowiedzia: boolean;
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
	selectedInspectionId: string | null;
	flashInspectionId?: string | null;
	onSelectInspection: (inspectionId: string) => void;
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
	selectedInspectionId,
	flashInspectionId = null,
	onSelectInspection,
	onSortByColumn,
	onResizeColumn,
	onOpenAdvancedFilter,
	onFilterChange,
	footer,
}: InspectionsDataTableProps) {
	useEffect(() => {
		if (!selectedInspectionId || typeof document === "undefined") {
			return;
		}

		const rowElement = document.querySelector(
			`tr[data-inspection-id="${selectedInspectionId}"]`,
		) as HTMLTableRowElement | null;
		rowElement?.scrollIntoView({ block: "center", behavior: "smooth" });
	}, [selectedInspectionId]);

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
				<tbody>
					{rows.map((row) => {
						const isActive = selectedInspectionId === row.id;
						const isFlashing = flashInspectionId === row.id;

						return (
							<tr
								key={row.id}
								data-inspection-id={row.id}
								onClick={() => onSelectInspection(row.id)}
								className={`cursor-pointer border-slate-200 border-b transition-colors last:border-b-0 ${
									isFlashing || isActive
										? "bg-blue-100 text-slate-900 ring-1 ring-blue-300 ring-inset"
										: "bg-white text-slate-900 hover:bg-slate-50"
								}`}
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
										(column.key === "dataWplywuPisma" &&
											rowFlags?.brakDataWplywuPisma) ||
										(column.key === "dataPismaZOdpowiedzia" &&
											rowFlags?.brakDataPismaZOdpowiedzia);
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
												? "-"
												: normalizedValue === "" || normalizedValue === "brak"
												? "-"
												: rawValue;
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
									const isRecommendationDatesColumn =
										column.key === "dataZalecen";
									const verticalListItems = isTeamColumn
										? displayValue
											.split(/[;,]/)
											.map((item) => item.trim())
											.filter(Boolean)
										: isScopeColumn
											? displayValue
												.split(/[;,]/)
												.map((item) => item.trim())
												.filter(Boolean)
										: isRecommendationDatesColumn
											? displayValue
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
											: displayValue)
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
											{(isTeamColumn || isScopeColumn) && displayValue !== "-" ? (
												<div className="subtle-vertical-scroll max-h-28 w-full space-y-1 overflow-y-auto pr-1">
													{shouldUseNumberedList ? (
														<ol className="list-decimal space-y-1 pl-4">
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
											) : isRecommendationDatesColumn && displayValue !== "-" ? (
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
														<span className="italic text-slate-400">{displayValue}</span>
													) : (
														displayValue
													)}
												</div>
											) : isLongTextColumn ? (
												<div className="w-full whitespace-normal break-words leading-5">
													{shouldShowNotApplicable ? (
														<span className="italic text-slate-400">{displayValue}</span>
													) : (
														displayValue
													)}
												</div>
											) : (
												<div className="w-full whitespace-normal break-words leading-5">
													{shouldShowNotApplicable ? (
														<span className="italic text-slate-400">{displayValue}</span>
													) : (
														displayValue
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
