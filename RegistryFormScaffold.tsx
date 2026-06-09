import { X } from "lucide-react";
import type { ReactNode } from "react";

type RegistryFormScaffoldProps = {
	isOpen: boolean;
	title: string;
	subtitle?: string;
	onClose: () => void;
	onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
	children: ReactNode;
	headerNotices?: ReactNode;
	footerContent?: ReactNode;
	isContentReadOnly?: boolean;
	cancelLabel?: string | null;
	submitLabel: ReactNode;
	isSubmitDisabled?: boolean;
	submitButtonClassName?: string;
	maxWidthClassName?: string;
	closeOnBackdropClick?: boolean;
};

export function RegistryFormScaffold({
	isOpen,
	title,
	subtitle,
	onClose,
	onSubmit,
	children,
	headerNotices,
	footerContent,
	isContentReadOnly = false,
	cancelLabel = "Anuluj",
	submitLabel,
	isSubmitDisabled = false,
	submitButtonClassName,
	maxWidthClassName = "max-w-3xl",
	closeOnBackdropClick = true,
}: RegistryFormScaffoldProps) {
	if (!isOpen) {
		return null;
	}

	return (
		<div
			className="fixed inset-0 z-50 flex items-center justify-center p-4"
			onMouseDown={(event) => {
				if (closeOnBackdropClick && event.target === event.currentTarget) {
					onClose();
				}
			}}
		>
			<div
				aria-hidden="true"
				className="absolute inset-0 bg-slate-950/65"
				onClick={closeOnBackdropClick ? onClose : undefined}
			/>
			<div
				className={`relative z-10 flex max-h-[92vh] w-full ${maxWidthClassName} flex-col overflow-hidden rounded-2xl border border-slate-300 bg-white p-4 text-slate-900 shadow-[0_24px_56px_rgba(2,8,23,0.35)] sm:p-5`}
			>
				<div className="mb-4 flex items-center justify-between gap-3 border-slate-200 border-b pb-3">
					<div>
						<h3 className="font-semibold text-base text-slate-900">{title}</h3>
						{subtitle ? (
							<p className="mt-1 text-slate-600 text-sm">{subtitle}</p>
						) : null}
						{headerNotices}
					</div>
					<button
						type="button"
						onClick={onClose}
						className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-300 text-slate-600 transition-colors hover:bg-slate-100"
					>
						<X size={14} />
					</button>
				</div>

				<form className="flex min-h-0 flex-1 flex-col" onSubmit={onSubmit}>
					<div
						className={`subtle-vertical-scroll min-h-0 flex-1 overflow-y-auto pr-1 ${
							isContentReadOnly ? "opacity-80" : ""
						}`}
					>
						{children}
					</div>

					<div className="sticky bottom-0 z-10 mt-3 border-slate-200 border-t bg-white pt-3">
						{footerContent}
						<div className="flex flex-wrap justify-end gap-2">
							{cancelLabel !== null ? (
								<button
									type="button"
									onClick={onClose}
									className="inline-flex h-9 items-center rounded-lg border border-slate-300 bg-transparent px-3 font-normal text-slate-700 text-sm transition-colors hover:bg-slate-100"
								>
									{cancelLabel}
								</button>
							) : null}

							<button
								type="submit"
								disabled={isSubmitDisabled}
								className={
									submitButtonClassName ??
									"inline-flex h-9 items-center rounded-lg border border-[#6ea3f0] bg-[#2d4d7f] px-3 font-normal text-slate-100 text-sm transition-colors hover:bg-[#375f99] disabled:cursor-not-allowed disabled:opacity-70"
								}
							>
								{submitLabel}
							</button>
						</div>
					</div>
				</form>
			</div>
		</div>
	);
}
