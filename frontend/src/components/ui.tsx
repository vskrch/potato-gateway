/**
 * Potato UI facade.
 *
 * Re-exports shadcn/ui primitives from ./components/ui/* and retains the
 * Potato-specific composite components (StatBox, StatusDot, EmptyState,
 * ErrorState, OfflineBanner, Spinner, CopyButton, CodeBlock, Toast) that have
 * no direct shadcn equivalent. Existing page imports (`from '../components/ui'`)
 * keep working unchanged; new code should prefer the granular shadcn imports
 * (`from '../components/ui/button'`).
 */
import React, { type ReactNode, useState, useEffect, useRef } from 'react'
import { Check, Copy, X, AlertTriangle, RefreshCw } from 'lucide-react'
import { cn } from '../lib/utils'

// shadcn primitives — re-exported for backward compatibility
export { Button } from './ui/button'
export type { ButtonProps } from './ui/button'
export { Input } from './ui/input'
export { Textarea } from './ui/textarea'
export { Select } from './ui/select'
export { Card, CardHeader, CardTitle, CardDescription, CardBody, CardContent, CardFooter } from './ui/card'
export { Badge } from './ui/badge'
export type { BadgeProps } from './ui/badge'
export { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription, DialogClose } from './ui/dialog'
export { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs'
export { Switch } from './ui/switch'
export { Slider } from './ui/slider'
export { Popover, PopoverTrigger, PopoverContent, PopoverAnchor } from './ui/popover'
export { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuShortcut, DropdownMenuGroup } from './ui/dropdown-menu'
export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from './ui/tooltip'
export { Label } from './ui/label'
export { Separator } from './ui/separator'
export { ScrollArea, ScrollBar } from './ui/scroll-area'
export { SkeletonPrimitive } from './ui/skeleton'
export { Toaster, toast } from './ui/sonner'

// Re-import shadcn primitives used by the composite components below
import { Button as ShadButton } from './ui/button'
import { Card as ShadCard } from './ui/card'
import { CardBody as ShadCardBody } from './ui/card'
import {
  Dialog as ShadDialog,
  DialogContent as ShadDialogContent,
  DialogHeader as ShadDialogHeader,
  DialogTitle as ShadDialogTitle,
} from './ui/dialog'

export function StatusDot({ ok, pulse = true }: { ok: boolean; pulse?: boolean }) {
  return (
    <span className="relative flex h-2 w-2 shrink-0">
      {pulse && (
        <span
          className={cn(
            'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
            ok ? 'bg-emerald-400' : 'bg-rose-400',
          )}
        />
      )}
      <span
        className={cn(
          'relative inline-flex rounded-full h-2 w-2',
          ok ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]',
        )}
      />
    </span>
  )
}

export function StatBox({
  label,
  value,
  sub,
  color,
  icon: Icon,
  trend,
  accent = 'violet',
}: {
  label: string
  value: string | number
  sub?: string
  color?: string
  icon?: React.ComponentType<{ className?: string }>
  trend?: { value: string; positive?: boolean }
  accent?: 'violet' | 'emerald' | 'amber' | 'cyan' | 'rose'
}) {
  const accentGradients = {
    violet: 'from-violet-500/80 via-fuchsia-500/40 to-transparent',
    emerald: 'from-emerald-500/80 via-teal-500/40 to-transparent',
    amber: 'from-amber-500/80 via-orange-500/40 to-transparent',
    cyan: 'from-cyan-500/80 via-blue-500/40 to-transparent',
    rose: 'from-rose-500/80 via-pink-500/40 to-transparent',
  }

  const iconGlows = {
    violet: 'group-hover:text-violet-300 group-hover:bg-violet-500/15 group-hover:border-violet-500/30',
    emerald: 'group-hover:text-emerald-300 group-hover:bg-emerald-500/15 group-hover:border-emerald-500/30',
    amber: 'group-hover:text-amber-300 group-hover:bg-amber-500/15 group-hover:border-amber-500/30',
    cyan: 'group-hover:text-cyan-300 group-hover:bg-cyan-500/15 group-hover:border-cyan-500/30',
    rose: 'group-hover:text-rose-300 group-hover:bg-rose-500/15 group-hover:border-rose-500/30',
  }

  return (
    <ShadCard className="p-5 flex flex-col justify-between relative overflow-hidden group hover:border-white/[0.18] transition-all duration-300 shadow-[0_4px_24px_rgba(0,0,0,0.3)] hover:shadow-[0_8px_32px_rgba(0,0,0,0.4)] hover:-translate-y-0.5">
      {/* Top ambient accent glow bar */}
      <div className={cn('absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r opacity-60 group-hover:opacity-100 transition-opacity', accentGradients[accent])} />
      
      {/* Background ambient radial blur */}
      <div className="absolute -top-12 -right-12 w-32 h-32 bg-gradient-to-br from-violet-500/10 via-fuchsia-500/5 to-transparent rounded-full blur-2xl group-hover:scale-150 transition-transform duration-500 pointer-events-none" />

      <div className="flex items-center justify-between gap-2 mb-3 relative z-10">
        <span className="text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">{label}</span>
        {Icon && (
          <div className={cn('p-2 rounded-xl bg-white/[0.04] border border-white/[0.07] text-zinc-400 transition-all duration-200', iconGlows[accent])}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-baseline justify-between gap-2 relative z-10">
        <span className={cn('text-2xl font-bold tracking-tight font-mono', color || 'text-white')}>{value}</span>
        {trend && (
          <span
            className={cn(
              'text-xs font-semibold px-2 py-0.5 rounded-md border font-mono',
              trend.positive
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/20',
            )}
          >
            {trend.value}
          </span>
        )}
      </div>

      {sub && <span className="text-xs text-zinc-400 mt-2 font-medium relative z-10">{sub}</span>}
    </ShadCard>
  )
}

export function QuickCopyPill({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }
  return (
    <button
      onClick={copy}
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-900/80 hover:bg-zinc-800 border border-white/[0.08] hover:border-violet-500/30 text-xs text-zinc-300 font-mono transition-all group"
      title="Click to copy"
    >
      {label && <span className="text-zinc-400 font-sans text-[11px]">{label}:</span>}
      <span className="text-violet-300 group-hover:text-white transition-colors">{text}</span>
      {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-zinc-500 group-hover:text-zinc-300" />}
    </button>
  )
}

export function ResilienceBadge({
  status = 'operational',
  sla = '99.99%',
}: {
  status?: 'operational' | 'degraded' | 'recovering'
  sla?: string
}) {
  const isOk = status === 'operational'
  return (
    <div className={cn(
      'inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono transition-all',
      isOk
        ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.12)]'
        : 'bg-amber-500/10 border-amber-500/25 text-amber-300'
    )}>
      <span className="relative flex h-2 w-2">
        {isOk && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />}
        <span className={cn('relative inline-flex rounded-full h-2 w-2', isOk ? 'bg-emerald-500' : 'bg-amber-500')} />
      </span>
      <span className="font-semibold text-white tracking-wide">SLA {sla}</span>
      <span className="text-[10px] text-zinc-400 uppercase">Resilience Active</span>
    </div>
  )
}

/**
 * Legacy single-toast component — kept for App.tsx which renders the queue
 * from useToastQueue. New code should use the `toast` export (sonner) instead.
 */
export function Toast({
  message,
  type,
  onDismiss,
  duration = 5000,
}: {
  message: string
  type: 'ok' | 'err'
  onDismiss: () => void
  duration?: number
}) {
  useEffect(() => {
    if (!message || !duration) return
    const t = setTimeout(onDismiss, duration)
    return () => clearTimeout(t)
  }, [message, duration, onDismiss])

  if (!message) return null
  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed bottom-6 right-6 bg-zinc-900/95 backdrop-blur-xl border border-white/[0.12] px-5 py-3.5 rounded-2xl text-[13px] z-[100] flex items-center gap-3 shadow-[0_20px_50px_rgba(0,0,0,0.6)] animate-[fadeIn_0.25s_ease-out] max-w-[calc(100vw-3rem)]"
    >
      <StatusDot ok={type === 'ok'} />
      <span className="text-zinc-100 font-medium">{message}</span>
      <button
        onClick={onDismiss}
        className="ml-2 text-zinc-500 hover:text-white transition-colors shrink-0"
        aria-label="Dismiss notification"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

export function Skeleton({ className, lines = 1, cards = 0 }: { className?: string; lines?: number; cards?: number }) {
  if (cards > 0) {
    return (
      <div className={cn('grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 animate-pulse', className)}>
        {Array.from({ length: cards }).map((_, i) => (
          <div key={i} className="bg-zinc-900/60 backdrop-blur-xl border border-white/[0.08] rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="h-2.5 w-24 bg-white/[0.08] rounded-full" />
              <div className="w-8 h-8 rounded-xl bg-white/[0.06]" />
            </div>
            <div className="h-7 w-20 bg-white/[0.08] rounded-full" />
            <div className="h-2.5 w-32 bg-white/[0.06] rounded-full" />
          </div>
        ))}
      </div>
    )
  }
  return (
    <div className={cn('space-y-2 animate-pulse', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="h-2.5 bg-white/[0.08] rounded-full" style={{ width: `${60 + (i % 3) * 20}%` }} />
      ))}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, children }: { icon?: React.ComponentType<{ className?: string }>; title: string; children?: ReactNode }) {
  return (
    <div className="p-8 sm:p-12 text-center text-zinc-500 text-xs flex flex-col items-center gap-3">
      {Icon ? <Icon className="w-8 h-8 text-zinc-600 stroke-1" /> : null}
      <div>
        <h3 className="text-sm font-semibold text-zinc-300 mb-1">{title}</h3>
        {children ? <div className="text-zinc-500">{children}</div> : null}
      </div>
    </div>
  )
}

export function ErrorState({ title, message, onRetry }: { title?: string; message?: string; onRetry?: () => void }) {
  return (
    <div className="p-6 sm:p-8 text-center text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-2xl flex flex-col items-center gap-3 text-xs">
      <AlertTriangle className="w-6 h-6 text-rose-400" />
      <div>
        <h3 className="text-sm font-semibold text-rose-200 mb-1">{title || 'Failed to load'}</h3>
        {message ? <p className="text-rose-300/80 max-w-md">{message}</p> : null}
      </div>
      {onRetry && (
        <ShadButton size="sm" variant="secondary" onClick={onRetry}>
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Try Again</span>
        </ShadButton>
      )}
    </div>
  )
}

export function OfflineBanner() {
  const [online, setOnline] = useState(navigator.onLine)
  useEffect(() => {
    const on = () => setOnline(true)
    const off = () => setOnline(false)
    window.addEventListener('online', on)
    window.addEventListener('offline', off)
    return () => {
      window.removeEventListener('online', on)
      window.removeEventListener('offline', off)
    }
  }, [])
  if (online) return null
  return (
    <div
      role="alert"
      className="fixed top-0 inset-x-0 z-[200] bg-amber-500/90 text-amber-950 px-4 py-2 text-xs font-semibold text-center shadow-lg backdrop-blur-sm"
    >
      You are offline. Some features may be unavailable until the connection is restored.
    </div>
  )
}

export function Spinner() {
  return (
    <div className="flex items-center justify-center py-16">
      <div className="relative w-8 h-8">
        <div className="w-8 h-8 border-2 border-violet-500/20 border-t-violet-500 rounded-full animate-spin" />
        <div className="absolute inset-0 w-8 h-8 border-2 border-fuchsia-500/10 border-b-fuchsia-500 rounded-full animate-spin [animation-duration:1.5s]" />
      </div>
    </div>
  )
}

/**
 * Legacy Modal — kept for pages that use the isOpen/onClose/title/children API.
 * Internally backed by shadcn Dialog (Radix) — a11y, focus-trap, scroll-lock
 * come for free. New code should use Dialog* directly for full control.
 */
export function Modal({
  isOpen,
  onClose,
  title,
  children,
}: {
  isOpen: boolean
  onClose: () => void
  title: string
  children: ReactNode
}) {
  // ponytail: delegate to shadcn Dialog (Radix) — gives us focus-trap + ESC +
  // scroll-lock for free. We adapt the isOpen/onClose API to Radix's
  // open/onOpenChange model. No manual focus management needed.
  return (
    <ShadDialog open={isOpen} onOpenChange={(o) => !o && onClose()}>
      <ShadDialogContent>
        <ShadDialogHeader>
          <ShadDialogTitle>{title}</ShadDialogTitle>
        </ShadDialogHeader>
        <div className="p-6 overflow-y-auto">{children}</div>
      </ShadDialogContent>
    </ShadDialog>
  )
}

export function CopyButton({ text, className }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className={cn(
        'p-1.5 rounded-lg border border-white/[0.08] bg-white/[0.04] text-zinc-400 hover:text-white hover:bg-white/[0.08] transition-all active:scale-95',
        className,
      )}
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  )
}

export function CodeBlock({ code, language = 'bash' }: { code: string; language?: string }) {
  return (
    <div className="relative bg-zinc-950 border border-white/[0.08] rounded-xl p-4 font-mono text-xs text-zinc-200 overflow-x-auto group">
      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <CopyButton text={code} />
      </div>
      {language && <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2 font-semibold select-none">{language}</div>}
      <pre className="whitespace-pre-wrap break-all">{code}</pre>
    </div>
  )
}