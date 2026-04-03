import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:   "bg-blue-600 text-white hover:bg-blue-700 shadow-sm",
        secondary: "bg-slate-100 text-slate-900 hover:bg-slate-200",
        outline:   "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
        ghost:     "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
        danger:    "bg-rose-600 text-white hover:bg-rose-700 shadow-sm",
        success:   "bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm",
        warning:   "bg-amber-500 text-white hover:bg-amber-600 shadow-sm",
      },
      size: {
        sm:   "h-8  px-3 text-xs",
        md:   "h-9  px-4",
        lg:   "h-10 px-5 text-base",
        icon: "h-9  w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
