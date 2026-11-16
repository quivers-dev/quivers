;; QIEC distribution bridge for generated Church programs. A QIEC tensor is
;; a tuple record until it reaches a distribution, where it becomes a
;; plain nested list; a log density is the runtime's `dist-score` at the
;; converted value, Booleans scoring as the numbers `flip` expects.
(define (_qvr-qiec-array value)
  (let ((value (_qvr-qiec-value value)))
    (cond
      ((and (pair? value) (equal? (_qvr-qiec-get value "qiec" #f) "tuple"))
       (map _qvr-qiec-array (_qvr-qiec-get value "items" '())))
      ((list? value) (map _qvr-qiec-array value))
      ((eq? value #t) 1)
      ((eq? value #f) 0)
      (else value))))
(define (_qvr-qiec-log-density distribution value)
  (dist-score distribution (_qvr-qiec-array value)))
(define (_qvr-qiec-log-mass distribution retained shape)
  ;; Every mapped Church distribution is normalized. A raw measure
  ;; needs its own checked mapping before this adapter can consume it.
  (if (or (< retained 0) (> retained (length shape)) (not (integer? retained)))
      (error "log_mass retained axes must be within the batch rank")
      (let build ((axis 0) (sizes shape))
        (if (= axis retained)
            0
            (map (lambda (unused) (build (+ axis 1) (cdr sizes)))
                 (iota (car sizes)))))))
;; Site labels a helper's draws and scores carry, counted so the n-th
;; occurrence of a label in one run of the model is "<label>@<n>", as
;; the reference machine replays them.
(define (_qvr-qiec-site-names)
  (let ((occurrences '()))
    (lambda (label)
      (let* ((entry (assoc label occurrences))
             (count (if entry (cdr entry) 0)))
        (set! occurrences (cons (cons label (+ count 1)) occurrences))
        (if (= count 0)
            label
            (string-append label "@" (number->string count)))))))
;; A helper's draw and scored weight under the runtime's own trace
;; primitives, each carrying its site name so a driver that clamps
;; sites by name can rebind these two alone.
(define (_qvr-qiec-draw label distribution) (sample distribution))
(define (_qvr-qiec-add label weight) (factor weight))
;; The program's canonical instances handled by the runtime's own
;; primitives, so a computation the model calls draws and scores as
;; the model does.
(define (_qvr-qiec-native-operations random-instance sample-operation score-instance add-operation)
  (let ((name (_qvr-qiec-site-names)))
    (list
      (cons (cons random-instance sample-operation)
            (lambda (request)
              (let* ((arguments (_qvr-qiec-get request "arguments"))
                     (label (car arguments))
                     (distribution (cadr arguments)))
                (_qvr-qiec-draw (name label) distribution))))
      (cons (cons score-instance add-operation)
            (lambda (request)
              (let ((weight (car (_qvr-qiec-get request "arguments"))))
                (_qvr-qiec-add (name "score") (_qvr-qiec-array weight))
                '()))))))
