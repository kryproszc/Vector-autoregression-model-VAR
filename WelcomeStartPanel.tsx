type WelcomeStartPanelProps = {
	operatorLogin: string;
};

export function WelcomeStartPanel({ operatorLogin: _operatorLogin }: WelcomeStartPanelProps) {
	return (
		<section className="flex min-h-[calc(100vh-8rem)] w-full items-center justify-center py-4">
			<div className="mx-auto w-full max-w-4xl rounded-3xl border border-[#2f4a75] bg-linear-to-br from-[#12284a] via-[#102441] to-[#0d1c35] p-8 text-center shadow-[0_26px_70px_rgba(2,8,23,0.45)] md:p-12">
				<h2 className="font-bold text-4xl text-slate-100 leading-tight md:text-5xl">
					Będzie dashboard
				</h2>
			</div>
		</section>
	);
}
